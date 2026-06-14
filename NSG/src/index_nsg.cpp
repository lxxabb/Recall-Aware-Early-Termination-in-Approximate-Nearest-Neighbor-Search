#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

namespace efanna2e {
#define _CONTROL_NUM 100

    IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                       Index *initializer)
            : Index(dimension, n, m), initializer_{initializer} {}

    IndexNSG::~IndexNSG() {
        if (distance_ != nullptr) {
            delete distance_;
            distance_ = nullptr;
        }
        if (initializer_ != nullptr) {
            delete initializer_;
            initializer_ = nullptr;
        }
        if (opt_graph_ != nullptr) {
            delete opt_graph_;
            opt_graph_ = nullptr;
        }
    }

    void IndexNSG::Save(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(final_graph_.size() == nd_);

        out.write((char *) &width, sizeof(unsigned));
        out.write((char *) &ep_, sizeof(unsigned));
        for (unsigned i = 0; i < nd_; i++) {
            unsigned GK = (unsigned) final_graph_[i].size();
            out.write((char *) &GK, sizeof(unsigned));
            out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
        }
        out.close();
    }
    void IndexNSG::GetIndegree()
    {
        indegree.assign(nd_, 0);
        hop_Count.assign(nd_, 0);
        //提取特征和进行预测时，需要使用如下两个特征，顺便在初始化入度时，初始化这个来年各个两个
        for (unsigned i = 0; i < nd_; i++)
        {
            _mm_prefetch(opt_graph_ + node_size * i + data_len, _MM_HINT_T0);
            unsigned* neighbors = (unsigned*)(opt_graph_ + node_size * i + data_len);
            unsigned MaxM = *neighbors;
            neighbors++;
            //此时neighbors指向第一个数据点，MaxM大小是该数据点的所有邻居数
            for (unsigned m = 0; m < MaxM; ++m)
            {
                _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                unsigned neighbor_id = neighbors[m];
                indegree[neighbor_id]++; // 邻居节点的入度加1
            }
        }
    }

    void IndexNSG::Load(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        in.read((char *) &width, sizeof(unsigned));
        in.read((char *) &ep_, sizeof(unsigned));
        // width=100;
        unsigned cc = 0;
        while (!in.eof()) {
            unsigned k;
            in.read((char *) &k, sizeof(unsigned));
            if (in.eof()) break;
            cc += k;
            std::vector<unsigned> tmp(k);
            in.read((char *) tmp.data(), k * sizeof(unsigned));
            final_graph_.push_back(tmp);
        }
        cc /= nd_;
        // std::cout<<cc<<std::endl;
    }

    void IndexNSG::Load_nn_graph(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        unsigned k;
        in.read((char *) &k, sizeof(unsigned));
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        size_t num = (unsigned) (fsize / (k + 1) / 4);
        in.seekg(0, std::ios::beg);

        final_graph_.resize(num);
        final_graph_.reserve(num);
        unsigned kk = (k + 3) / 4 * 4;
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur);
            final_graph_[i].resize(k);
            final_graph_[i].reserve(kk);
            in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
        }
        in.close();
    }

    void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        boost::dynamic_bitset<> flags{nd_, 0};
        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
            init_ids[i] = final_graph_[ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                            (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = distance_->compare(query, data_ + dimension_ * (size_t) id,
                                                    (unsigned) dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                                 boost::dynamic_bitset<> &flags,
                                 std::vector<Neighbor> &retset,
                                 std::vector<Neighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
            init_ids[i] = final_graph_[ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            // std::cout<<id<<std::endl;
            float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                            (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            fullset.push_back(retset[i]);
            // flags[id] = 1;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float dist = distance_->compare(query, data_ + dimension_ * (size_t) id,
                                                    (unsigned) dimension_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance) continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void IndexNSG::init_graph(const Parameters &parameters) {
        float *center = new float[dimension_];
        for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
        for (unsigned i = 0; i < nd_; i++) {
            for (unsigned j = 0; j < dimension_; j++) {
                center[j] += data_[i * dimension_ + j];
            }
        }
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] /= nd_;
        }
        std::vector<Neighbor> tmp, pool;
        ep_ = rand() % nd_;  // random initialize navigating point
        get_neighbors(center, parameters, tmp, pool);
        ep_ = tmp[0].id;
        delete center;
    }

    void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                              const Parameters &parameter,
                              boost::dynamic_bitset<> &flags,
                              SimpleNeighbor *cut_graph_) {
        unsigned range = parameter.Get<unsigned>("R");
        unsigned maxc = parameter.Get<unsigned>("C");
        width = range;
        unsigned start = 0;

        for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
            unsigned id = final_graph_[q][nn];
            if (flags[id]) continue;
            float dist =
                    distance_->compare(data_ + dimension_ * (size_t) q,
                                       data_ + dimension_ * (size_t) id, (unsigned) dimension_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);

        while (result.size() < range && (++start) < pool.size() && start < maxc) {
            auto &p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = distance_->compare(data_ + dimension_ * (size_t) result[t].id,
                                               data_ + dimension_ * (size_t) p.id,
                                               (unsigned) dimension_);
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(p);
        }

        SimpleNeighbor *des_pool = cut_graph_ + (size_t) q * (size_t) range;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
            des_pool[result.size()].distance = -1;
        }
    }

    void IndexNSG::InterInsert(unsigned n, unsigned range,
                               std::vector<std::mutex> &locks,
                               SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1) break;

            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++) {
                    if (des_pool[j].distance == -1) break;
                    if (n == des_pool[j].id) {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup) continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = distance_->compare(data_ + dimension_ * (size_t) result[t].id,
                                                       data_ + dimension_ * (size_t) p.id,
                                                       (unsigned) dimension_);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range) des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        /*
        std::cout << " graph link" << std::endl;
        unsigned progress=0;
        unsigned percent = 100;
        unsigned step_size = nd_/percent;
        std::mutex progress_lock;
        */
        unsigned range = parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(nd_);

#pragma omp parallel
        {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
            boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
                pool.clear();
                tmp.clear();
                flags.reset();
                get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
                sync_prune(n, pool, parameters, flags, cut_graph_);
                /*
              cnt++;
              if(cnt % step_size == 0){
                LockGuard g(progress_lock);
                std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
                }
                */
            }
        }

#pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; ++n) {
            InterInsert(n, range, locks, cut_graph_);
        }
    }

    void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
        std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
        unsigned range = parameters.Get<unsigned>("R");
        Load_nn_graph(nn_graph_path.c_str());
        data_ = data;
        init_graph(parameters);
        SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t) range];
        Link(parameters, cut_graph_);
        final_graph_.resize(nd_);

        for (size_t i = 0; i < nd_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t) range;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < range; j++) {
                if (pool[j].distance == -1) break;
                pool_size = j;
            }
            pool_size++;
            final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                final_graph_[i][j] = pool[j].id;
            }
        }

        tree_grow(parameters);

        unsigned max = 0, min = 1e6, avg = 0;
        for (size_t i = 0; i < nd_; i++) {
            auto size = final_graph_[i].size();
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= 1.0 * nd_;
        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

        has_built = true;
        delete cut_graph_;
    }

    void IndexNSG::Search(const float *query, const float *x, size_t K,
                          const Parameters &parameters, unsigned *indices) {
        const unsigned L = parameters.Get<unsigned>("L_search");
        data_ = x;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
            init_ids[tmp_l] = final_graph_[ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
                    distance_->compare(data_ + dimension_ * id, query, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            // flags[id] = true;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float dist =
                            distance_->compare(query, data_ + dimension_ * id, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                      const Parameters &parameters, unsigned *indices) {
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_get_every_points_recall(const float *query, size_t K, unsigned i,
                                      const Parameters &parameters, unsigned *indices, const char *every_points) {
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
        float recall_raw = calculate_precision(i, retset, K);
        csvFile << i << ","
                << 0 << ","
                << recall_raw << ","
                << search_time_ms << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Laet_Darth_train_data(const float *query, size_t K,
                                                      const Parameters &parameters, unsigned *indices,
                                                      unsigned i, const char *train_feature_filename) {
        unsigned distance_count=0;
        unsigned step=0;
        unsigned inserts=0;
        //保证准确率为1时，最多收集20次，不为1时最多收集10次
        std::unordered_map<int, int> recall_cnt;
        //训练集特征位置
        std::ofstream csvFile(train_feature_filename, std::ios::app);

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_count++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;
        int visited_points=0;
        int duration=0;
        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_count++;
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //前K个数据点的平均距离和方差
                    double sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    //分位数
                    double percentile_25=retset[K*0.25].distance;
                    double percentile_50=retset[K*0.5].distance;
                    double percentile_75=retset[K*0.75].distance;
                    //准确率
                    float recall_raw = calculate_precision(i, retset, K);
                    // 保留 2 位小数（数学意义上的）
                    float recall_2 = std::round(recall_raw * 100.0f) / 100.0f;
                    // 映射为整数 key
                    int recall_key = static_cast<int>(recall_2 * 100);
                    // 每个 recall 的最大采样次数
                    int max_collect = (recall_key == 100) ? 20 : 10;
                    // 是否还能采样
                    if (recall_cnt[recall_key] < max_collect) {
                        recall_cnt[recall_key]++;
                        csvFile << i << ","
                                << step << ","
                                << distance_count << ","
                                << inserts << ","
                                << visited_points << ","
                                << duration << ","
                                << first_nn_dist << ","
                                << nn_dist << ","
                                << furthest_dist << ","
                                << nn10_dist << ","
                                << nn_to_first << ","
                                << nn10_to_first << ","
                                << mean_distance << ","
                                << variance_of_distances << ","
                                << percentile_25 << ","
                                << percentile_50 << ","
                                << percentile_75 << ","
                                << recall_2 << "\n";
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
            step++;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_laet_test(const float *query, size_t K,unsigned i,
                                      const Parameters &parameters, unsigned *indices,unsigned laet_F,float laet_prediction_multiplier,BoosterHandle booster,const char *every_points) {
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        unsigned distance_count=0;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_count++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;
        float first_nn_dist=retset[0].distance;
        int ndis_target = std::numeric_limits<int>::max();
        bool flag_stop=false;
        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_count++;
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                    if(distance_count==laet_F){
                        model_pre_cout++;
                        auto start_1 = std::chrono::high_resolution_clock::now();
                        unsigned feat_num=5;
                        int num_feats = feat_num + dimension_;
                        std::vector<double> features(num_feats); // 假设特征类型为 double
                        std::vector<double> output(1); // 假设输出类型为 double
                        float nn_dist=retset[0].distance;
                        float nn10_dist=retset[9].distance;
                        float nn_to_first=nn_dist/first_nn_dist;
                        float nn10_to_first=nn10_dist/first_nn_dist;
                        features[0] = first_nn_dist;
                        features[1] = nn_dist;
                        features[2] = nn10_dist;
                        features[3] = nn_to_first;
                        features[4] = nn10_to_first;
                        for (int i = 0; i < dimension_; ++i) {
                            features[i + feat_num] = static_cast<double>(query[i]);
                        }

                        double out_result[1];
                        long int out_len;
                        int res = LGBM_BoosterPredictForMatSingleRow(
                                booster,
                                features.data(),
                                C_API_DTYPE_FLOAT64,
                                num_feats,
                                1,
                                C_API_PREDICT_NORMAL,
                                0,
                                -1,
                                "",
                                &out_len,
                                out_result);
                        ndis_target = (int)out_result[0];
                        ndis_target = static_cast<int>(std::round(laet_prediction_multiplier * ndis_target));
                        //记录模型调用时间消耗
                        auto end_1 = std::chrono::high_resolution_clock::now();
                        time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    }
                    if(distance_count>=ndis_target){
                        flag_stop=true;
                        break;
                    }
                }
                if(flag_stop){
                    break;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

        float recall_raw = calculate_precision(i, retset, K);
        csvFile << i << ","
                << 0 << ","
                << recall_raw << ","
                << search_time_ms << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Darth_test_time(const float *query, size_t K,
                                                const Parameters &parameters, unsigned *indices,unsigned i,unsigned ipi,unsigned mpi,float target_recall,BoosterHandle booster,const char *every_points) {
//        std::ofstream csvFile("/root/NSG-data/results/darth/SIFT1M/k100/每个数据点搜索情况_{target_recall}.csv", std::ios::out | std::ios::app);
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();
        unsigned distance_count=0;
        unsigned step=0;
        unsigned inserts=0;
        unsigned prediction_interval=ipi;
        unsigned distance_interval=0;
        bool flag_stop=false;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_count++;
            distance_interval++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        float predicted_recall=0.0;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_count++;
                    distance_interval++;
                    //判断是否进行模型预测
                    if(distance_interval % prediction_interval==0){
                        model_pre_cout++;
                        //收集特征
                        float nn_dist=retset[0].distance;
                        float furthest_dist=retset[K-1].distance;
                        //前K个数据点的平均距离和方差
                        double sum = 0.0;
                        for (int i = 0; i < K; ++i) {
                            sum += retset[i].distance;
                        }
                        double mean_distance = sum / K; // 平均距离
                        // 2. 计算方差（整数部分）
                        double sq_sum = 0.0;
                        for (int i = 0; i < K; ++i) {
                            double diff = retset[i].distance - mean_distance;
                            sq_sum += diff * diff; // 避免用pow()提高效率
                        }
                        double variance_of_distances = sq_sum / K;
                        //分位数
                        double percentile_25=retset[K*0.25].distance;
                        double percentile_50=retset[K*0.5].distance;
                        double percentile_75=retset[K*0.75].distance;

                        unsigned feat_num=11;
                        int num_feats = feat_num ;
                        std::vector<double> features(num_feats); // 假设特征类型为 double
                        std::vector<double> output(1); // 假设输出类型为 double

                        features[0] = step;
                        features[1] = distance_count;
                        features[2] = inserts;
                        features[3] = first_nn_dist;
                        features[4] = nn_dist;
                        features[5] = furthest_dist;
                        features[6] = mean_distance;
                        features[7] = variance_of_distances;
                        features[8] = percentile_25;
                        features[9] = percentile_50;
                        features[10] = percentile_75;

                        double out_result[1];
                        long int out_len;
                        int res = LGBM_BoosterPredictForMatSingleRow(
                                booster,
                                features.data(),
                                C_API_DTYPE_FLOAT64,
                                num_feats,
                                1,
                                C_API_PREDICT_NORMAL,
                                0,
                                -1,
                                "",
                                &out_len,
                                out_result);
                        predicted_recall = (float) out_result[0];
                        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

                        if (predicted_recall >= target_recall) {
//                            float recall_raw = calculate_precision(i, retset, K);
                            flag_stop=true;
                            break;
                        }
                        float recall_diff = target_recall - predicted_recall;
                        prediction_interval = mpi + (ipi - mpi) * recall_diff;
                        distance_interval=0;

                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                }
                if(flag_stop){
                    break;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
            step++;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

        float recall_raw = calculate_precision(i, retset, K);
        csvFile << i << ","
                << predicted_recall << ","
                << recall_raw << ","
                << search_time_ms << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Darth_test_metrics(const float *query, size_t K,
                                                      const Parameters &parameters, unsigned *indices,unsigned i,unsigned ipi,unsigned mpi,float target_recall,BoosterHandle booster,unsigned& success_stop_count,unsigned& stop_count) {
        unsigned distance_count=0;
        unsigned step=0;
        unsigned inserts=0;
        unsigned prediction_interval=ipi;
        unsigned distance_interval=0;
        bool flag_stop=false;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_count++;
            distance_interval++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_count++;
                    distance_interval++;
                    //判断是否进行模型预测
                    if(distance_interval % prediction_interval==0){
                        model_pre_cout++;
                        auto start_1 = std::chrono::high_resolution_clock::now();
                        //收集特征
                        float nn_dist=retset[0].distance;
                        float furthest_dist=retset[K-1].distance;
                        //前K个数据点的平均距离和方差
                        double sum = 0.0;
                        for (int i = 0; i < K; ++i) {
                            sum += retset[i].distance;
                        }
                        double mean_distance = sum / K; // 平均距离
                        // 2. 计算方差（整数部分）
                        double sq_sum = 0.0;
                        for (int i = 0; i < K; ++i) {
                            double diff = retset[i].distance - mean_distance;
                            sq_sum += diff * diff; // 避免用pow()提高效率
                        }
                        double variance_of_distances = sq_sum / K;
                        //分位数
                        double percentile_25=retset[K*0.25].distance;
                        double percentile_50=retset[K*0.5].distance;
                        double percentile_75=retset[K*0.75].distance;

                        unsigned feat_num=11;
                        int num_feats = feat_num ;
                        std::vector<double> features(num_feats); // 假设特征类型为 double
                        std::vector<double> output(1); // 假设输出类型为 double

                        features[0] = step;
                        features[1] = distance_count;
                        features[2] = inserts;
                        features[3] = first_nn_dist;
                        features[4] = nn_dist;
                        features[5] = furthest_dist;
                        features[6] = mean_distance;
                        features[7] = variance_of_distances;
                        features[8] = percentile_25;
                        features[9] = percentile_50;
                        features[10] = percentile_75;

                        double out_result[1];
                        long int out_len;
                        int res = LGBM_BoosterPredictForMatSingleRow(
                                booster,
                                features.data(),
                                C_API_DTYPE_FLOAT64,
                                num_feats,
                                1,
                                C_API_PREDICT_NORMAL,
                                0,
                                -1,
                                "",
                                &out_len,
                                out_result);
                        float predicted_recall = (float) out_result[0];
                        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

                        //记录模型调用时间消耗
                        auto end_1 = std::chrono::high_resolution_clock::now();
                        time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                        if (predicted_recall >= target_recall) {
                            //真实准确率
                            float recall_raw = calculate_precision(i, retset, K);
                            if(recall_raw >= target_recall){
                                success_stop_count++;
                            }
                            flag_stop=true;
                            stop_count++;
                            break;
                        }
                        float recall_diff = target_recall - predicted_recall;
                        prediction_interval = mpi + (ipi - mpi) * recall_diff;
                        distance_interval=0;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                }
                if(flag_stop){
                    break;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
            step++;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Raet_recall_train_data(const float *query, size_t K,
                                                            const Parameters &parameters, unsigned *indices,
                                                            unsigned i, const char *train_feature_filename) {
        unsigned distance_count=0;
        unsigned step=0;
        unsigned inserts=0;
        //保证准确率为1时，最多收集20次，不为1时最多收集10次
        std::unordered_map<int, int> recall_cnt;
        //训练集特征位置
        std::ofstream csvFile(train_feature_filename, std::ios::app);

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_count++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;
                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_count++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (k > 0) {
                        float groundtruth = calculate_precision(i, retset, K);
                        bool groundtruth_flag = true;
//                        if (K == 100) {
//                            groundtruth_flag = groundtruth >= 0.9;
//                        } else if (K == 10) {
//                            groundtruth_flag = groundtruth >= 0.5;
//                        }
                        // if (groundtruth_flag && current_insert_flag &&!(groundtruth == 1.0f && full_accuracy_count >= 10)) {
                        if (groundtruth_flag && !(groundtruth == 1.0f && full_accuracy_count >= 20)) {
                            //收集特征
                            float nn_dist=retset[0].distance;
                            float nn10_dist=retset[9].distance;
                            float nn_to_first = nn_dist/first_nn_dist;
                            float nn10_to_first = nn10_dist/first_nn_dist;
                            float furthest_dist=retset[K-1].distance;
                            //当前CNS中已访问的数据点数量
                            unsigned visited_points = 0;
                            for (int i = 0; i < L; i++) {
                                if (retset[i].flag == false) {
                                    visited_points++;
                                }
                            }
                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            //分位数
                            double percentile_25=retset[K*0.25].distance;
                            double percentile_50=retset[K*0.5].distance;
                            double percentile_75=retset[K*0.75].distance;
                            //准确率
                            float recall_raw = calculate_precision(i, retset, K);
                            // 保留 2 位小数（数学意义上的）
                            float recall_2 = std::round(recall_raw * 100.0f) / 100.0f;
                            // 映射为整数 key
                            int recall_key = static_cast<int>(recall_2 * 100);
                            // 每个 recall 的最大采样次数
                            int max_collect = (recall_key == 100) ? 20 : 10;
                            // 是否还能采样（根据每一个数据点的每一个准确率最多有10条，准确率为1最多有20条）
//                            if (recall_cnt[recall_key] < max_collect) {
                                recall_cnt[recall_key]++;
                                csvFile << i << ","
                                        << step << ","
                                        << distance_count << ","
                                        << inserts << ","
                                        << visited_points << ","
                                        << first_nn_dist << ","
                                        << nn_dist << ","
                                        << furthest_dist << ","
                                        << nn10_dist << ","
                                        << nn_to_first << ","
                                        << nn10_to_first << ","
                                        << mean_distance << ","
                                        << variance_of_distances << ","
                                        << percentile_25 << ","
                                        << percentile_50 << ","
                                        << percentile_75 << ","
                                        << recall_2 << "\n";
//                            }

                            if (groundtruth == 1.0f) {
                                full_accuracy_count++;
                            }
                        }
                    }
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
            step++;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::SearchWithOptGraph_Raet_CNS_train_data(const float *query, size_t K,
                                                 const Parameters &parameters, unsigned *indices, const float acc_thold,
                                                 unsigned qid, const char *train_feature_filename) {

        //训练集特征位置
        std::ofstream csvFile(train_feature_filename, std::ios::app);
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1){
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }


                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];

                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t j = 0; j < dimension_; ++j) {
                            sim += query[j] * vec[j];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }
                    csvFile << qid << ","
                            << step << ","
                            << distance_counts << ","
                            << inserts << ","
                            << first_nn_dist << ","
                            << nn_dist << ","
                            << furthest_dist << ","
                            << mean_distance << ","
                            << variance_of_distances << ","
                            << avg_hop << ","
                            << avg_indegree << ","
                            << density << ","
                            << entry_query_dist_ratio << ","
                            << dist_distribution_entropy << ","
                            << skewness << ","
                            << kurtosis << ","
                            << energy << ","
                            << cosine_mean << ","
                            << cosine_variance << ","
                            << cosine_direction_entropy << "\n";
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_our_Raet_CNS_test_regression(const float *query, size_t K,
                                                          const Parameters &parameters, unsigned *indices, const float target_recall,
                                                          unsigned qid,BoosterHandle booster) {
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1){
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];


                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }

                    auto start_1 = std::chrono::high_resolution_clock::now();
                    unsigned feat_num=19;
                    int num_feats = feat_num + dimension_;
                    std::vector<double> features(num_feats); // 假设特征类型为 double
                    std::vector<double> output(1); // 假设输出类型为 double
                    features[0] = step;
                    features[1] = distance_counts;
                    features[2] = inserts;
                    features[3] = first_nn_dist;
                    features[4] = nn_dist;
                    features[5] = furthest_dist;
                    features[6] = mean_distance;
                    features[7] = variance_of_distances;
                    features[8] = avg_hop;
                    features[9] = avg_indegree;
                    features[10] = density;
                    features[11] = entry_query_dist_ratio;
                    features[12] = dist_distribution_entropy;
                    features[13] = skewness;
                    features[14] = kurtosis;
                    features[15] = energy;
                    features[16] = cosine_mean;
                    features[17] = cosine_variance;
                    features[18] = cosine_direction_entropy;

                    for (int i = 0; i < dimension_; ++i) {
                        features[i + feat_num] = static_cast<double>(query[i]);
                    }

                    double out_result[1];
                    long int out_len;
                    int res = LGBM_BoosterPredictForMatSingleRow(
                            booster,
                            features.data(),
                            C_API_DTYPE_FLOAT64,
                            num_feats,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            -1,
                            "",
                            &out_len,
                            out_result);
                    int predicted_CNS = out_result[0];
                    predicted_CNS = std::max((int)K, predicted_CNS);
                    predicted_CNS = std::min((int)L, predicted_CNS);
                    //记录模型调用时间消耗
                    auto end_1 = std::chrono::high_resolution_clock::now();
                    time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    retset.resize(predicted_CNS+1);
                    L=predicted_CNS;
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_our_Raet_CNS_test_regression_Noquery_Noske_Nodensty(const float *query, size_t K,
                                                                   const Parameters &parameters, unsigned *indices, const float target_recall,
                                                                   unsigned qid,BoosterHandle booster,const char *every_points) {
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();

        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;
        bool flag_model_predict=true;
        int k = 0;
        int predicted_CNS=0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1 && flag_model_predict){
                    model_pre_cout++;
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];


                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }

                    auto start_1 = std::chrono::high_resolution_clock::now();
                    unsigned feat_num=15;
                    int num_feats = feat_num ;
                    std::vector<double> features(num_feats); // 假设特征类型为 double
                    std::vector<double> output(1); // 假设输出类型为 double
                    features[0] = step;
                    features[1] = distance_counts;
                    features[2] = inserts;
                    features[3] = first_nn_dist;
                    features[4] = nn_dist;
                    features[5] = furthest_dist;
                    features[6] = mean_distance;
                    features[7] = variance_of_distances;
                    features[8] = avg_hop;
                    features[9] = avg_indegree;

                    features[10] = entry_query_dist_ratio;
                    features[11] = dist_distribution_entropy;

                    features[12] = cosine_mean;
                    features[13] = cosine_variance;
                    features[14] = cosine_direction_entropy;


                    double out_result[1];
                    long int out_len;
                    int res = LGBM_BoosterPredictForMatSingleRow(
                            booster,
                            features.data(),
                            C_API_DTYPE_FLOAT64,
                            num_feats,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            -1,
                            "",
                            &out_len,
                            out_result);
                    predicted_CNS = out_result[0];
                    predicted_CNS = std::max((int)K, predicted_CNS);
                    predicted_CNS = std::min((int)L, predicted_CNS);
                    //记录模型调用时间消耗
                    auto end_1 = std::chrono::high_resolution_clock::now();
                    time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    retset.resize(predicted_CNS+1);
                    L=predicted_CNS;
                    flag_model_predict=false;
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

        float recall_raw = calculate_precision(qid, retset, K);
        csvFile << qid << ","
                << predicted_CNS << ","
                << recall_raw << ","
                << search_time_ms << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_our_Raet_CNS_selector(const float *query, size_t K,
                                                         const Parameters &parameters, unsigned *indices,unsigned qid,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,
                                                         BoosterHandle booster_recall,BoosterHandle booster_CNS,const char *every_points) {
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_0_1.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_ge90.csv", std::ios::out | std::ios::app);
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();
        int recall_or_CNS_selector=0;//0选召回率方法，1选CNS方法
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        float predicted_recall=0;
        int visited_points=0;
        int duration=0;
        int k = 0;

        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;
        bool flag_model_predict=true;
        int predicted_CNS=0;

        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts==stability_times_r1){
                    int test=1;
                }

                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/K)*pre_now_stability_insert >= target_recall || first_flag || stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }

                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;
                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    current_insert_flag=true;

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict && recall_or_CNS_selector==0){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=9;
                            int num_feats = feat_num;
//                            int num_feats = feat_num+ dimension_;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = static_cast<float>(visited_points)/K;
                            features[4] = duration;
                            features[5] = nn_dist;
                            features[6] = furthest_dist;
                            features[7] = mean_distance;
                            features[8] = variance_of_distances;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster_recall,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;
                        }

                    }
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;

                entry_point_processed=false;
                //收集特征
                if(k==K-1 && flag_model_predict){
                    model_pre_cout++;
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];


                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }
                    //todo 进行阈值选择，使用召回率还是使用CNS方法，如果使用召回率方法，否则 执行CNS预测 继续执行下面代码并recall_or_CNS_selector=1.
                    //执行cns预测
                    //DEEP if(mean_distance>-0.65)执行CNS
//                    if(mean_distance>-0.65){
                        recall_or_CNS_selector=1;
                        auto start_1 = std::chrono::high_resolution_clock::now();
                        unsigned feat_num=15;
                        int num_feats = feat_num ;
                        std::vector<double> features(num_feats); // 假设特征类型为 double
                        std::vector<double> output(1); // 假设输出类型为 double
                        features[0] = step;
                        features[1] = distance_counts;
                        features[2] = inserts;
                        features[3] = first_nn_dist;
                        features[4] = nn_dist;
                        features[5] = furthest_dist;
                        features[6] = mean_distance;
                        features[7] = variance_of_distances;
                        features[8] = avg_hop;
                        features[9] = avg_indegree;

                        features[10] = entry_query_dist_ratio;
                        features[11] = dist_distribution_entropy;

                        features[12] = cosine_mean;
                        features[13] = cosine_variance;
                        features[14] = cosine_direction_entropy;


                        double out_result[1];
                        long int out_len;
                        int res = LGBM_BoosterPredictForMatSingleRow(
                                booster_CNS,
                                features.data(),
                                C_API_DTYPE_FLOAT64,
                                num_feats,
                                1,
                                C_API_PREDICT_NORMAL,
                                0,
                                -1,
                                "",
                                &out_len,
                                out_result);
                        predicted_CNS = out_result[0];
                        predicted_CNS = std::max((int)K, predicted_CNS);
                        predicted_CNS = std::min((int)L, predicted_CNS);
                        //记录模型调用时间消耗
                        auto end_1 = std::chrono::high_resolution_clock::now();
                        time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                        retset.resize(predicted_CNS+1);
                        L=predicted_CNS;
//                    }
                    flag_model_predict=false;
                }
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }

                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

        float recall_raw = calculate_precision(qid, retset, K);
        csvFile << qid << ","<< predicted_recall << ","<< recall_raw << ","<< search_time_ms << ","<< recall_or_CNS_selector << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Raet_test_regression_metrics(const float *query, size_t K,
                                                                   const Parameters &parameters,
                                                                   unsigned int *indices, unsigned int qid,
                                                                   const float target_recall,
                                                                   BoosterHandle booster,
                                                                   unsigned int &success_stop_count) {
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1){
                    auto start_1 = std::chrono::high_resolution_clock::now();
                    model_pre_cout++;
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];

                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }
                    unsigned feat_num=19;
                    int num_feats = feat_num + dimension_;
                    std::vector<double> features(num_feats); // 假设特征类型为 double
                    std::vector<double> output(1); // 假设输出类型为 double
                    features[0] = step;
                    features[1] = distance_counts;
                    features[2] = inserts;
                    features[3] = first_nn_dist;
                    features[4] = nn_dist;
                    features[5] = furthest_dist;
                    features[6] = mean_distance;
                    features[7] = variance_of_distances;
                    features[8] = avg_hop;
                    features[9] = avg_indegree;
                    features[10] = density;
                    features[11] = entry_query_dist_ratio;
                    features[12] = dist_distribution_entropy;
                    features[13] = skewness;
                    features[14] = kurtosis;
                    features[15] = energy;
                    features[16] = cosine_mean;
                    features[17] = cosine_variance;
                    features[18] = cosine_direction_entropy;

                    for (int i = 0; i < dimension_; ++i) {
                        features[i + feat_num] = static_cast<double>(query[i]);
                    }

                    double out_result[1];
                    long int out_len;
                    int res = LGBM_BoosterPredictForMatSingleRow(
                            booster,
                            features.data(),
                            C_API_DTYPE_FLOAT64,
                            num_feats,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            -1,
                            "",
                            &out_len,
                            out_result);
                    int predicted_CNS = out_result[0];
                    predicted_CNS = std::max((int)K, predicted_CNS);
                    predicted_CNS = std::min((int)L, predicted_CNS);
                    //记录模型调用时间消耗
                    auto end_1 = std::chrono::high_resolution_clock::now();
                    time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    retset.resize(predicted_CNS+1);
                    L=predicted_CNS;
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }


    void IndexNSG::SearchWithOptGraph_stablize(const float *query, size_t K,
                                               const Parameters &parameters, unsigned *indices, const float target_recall,
                                               unsigned i, int& all_stability_counts,const char *stablity_filename) {

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        std::ofstream csvFile(stablity_filename, std::ios::app);
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        //稳定次数
        int stability_counts = 0;
        int step=0;
        //结束标志
        bool stop_flag = false;
        bool flag_CNS_noindex0=false;
        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
//                    if (flag_CNS_noindex0) {
//                        float groundtruth = calculate_precision(i, retset, K);
//                        if (groundtruth >= target) {
//                            stop_flag = true;
//                            break;
//                        }
//                    }
                }//当前访问的CNS中的该数据点 的 所有邻居已经访问过
                if (stop_flag) {
                    break;
                }
                //计算稳定次数
                if(flag_CNS_noindex0){
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        if (current_insert_count*100 <= K * (100 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            stability_counts = 0;
                        }
                    }
                }
                float recall_raw = calculate_precision(i, retset, K);
                csvFile<< i << ","    // 当前 query ID
                        << step << ","        // 当前 step（或你想写的指标）
                        << stability_counts<< ","
                        << recall_raw<<"\n";
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        all_stability_counts+=stability_counts;
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::SearchWithOptGraph_pre_recall(const float *query, size_t K,
                                                 const Parameters &parameters, unsigned *indices, const float target_recall,
                                                 size_t stability_times,BoosterHandle booster) {

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        unsigned inserts=0;
        unsigned distance_counts=0;
        unsigned step=0;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;
        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            dist_cout++;
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int consecutive_jaccard_ones = 0;

        //退出外层循环标志
        bool stop_flag = false;

        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;
                //上一次模型预测结果
                double last_model_pre = 0;
                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    dist_cout++;
                    distance_counts++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        last_model_pre += 1.0/K;
                    }
//                    if(flag_CNS_noindex0){
                        //收集特征进行模型预测
                        if (consecutive_jaccard_ones >= stability_times && (last_model_pre>= target_recall || first_model_pre)){
                            model_pre_cout++;
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=8;
                            int num_feats = feat_num + dimension_;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;
                            //当前CNS中已访问的数据点数量
                            double visited_points = 0;
                            for (int i = 0; i < L; i++) {
                                if (retset[i].flag == false) {
                                    visited_points++;
                                }
                            }
                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = visited_points;
                            features[4] = nn_dist;
                            features[5] = furthest_dist;
                            features[6] = mean_distance;
                            features[7] = variance_of_distances;
                            for (int i = 0; i < dimension_; ++i) {
                                features[i + feat_num] = static_cast<double>(query[i]);
                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            double predicted_recall =std::max(out_result[0],last_model_pre);
//                            float predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            last_model_pre = predicted_recall;
                            first_model_pre = false;
                        }
//                    }
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
//                if (k>=1) {
                    // 如果连续 ”几次“ CNS相似度大于95，那么直接返回结果  30是0.98
                    //                    if(consecutive_jaccard_ones>=50){
                    //                        break;
                    //                    }
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (consecutive_jaccard_ones > 0 && current_insert_count == 0) {
                        consecutive_jaccard_ones++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target_recall)，即相似数量为 K*target_recall ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            consecutive_jaccard_ones++;
                        } else {
                            consecutive_jaccard_ones = 0;
                        }
                    }
//                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::SearchWithOptGraph_pre_recall_lessPre(const float *query, size_t K,unsigned i,
                                                         const Parameters &parameters, unsigned *indices,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,BoosterHandle booster,const char *every_points) {
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_0_1.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_ge90.csv", std::ios::out | std::ios::app);
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        float predicted_recall=0;
        int visited_points=0;
        int duration=0;
        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts==stability_times_r1){
                    int test=1;
                }

                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/K)*pre_now_stability_insert >= target_recall || first_flag || stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }

                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    current_insert_flag=true;

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=9;
                            int num_feats = feat_num;
//                            int num_feats = feat_num+ dimension_;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = static_cast<float>(visited_points)/K;
                            features[4] = duration;
                            features[5] = nn_dist;
                            features[6] = furthest_dist;
                            features[7] = mean_distance;
                            features[8] = variance_of_distances;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;
                        }

                    }
                }
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        double search_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

        float recall_raw = calculate_precision(i, retset, K);
        csvFile << i << ","<< predicted_recall << ","<< recall_raw << ","<< search_time_ms << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_pre_recall_seletor_label(const float *query, size_t K,unsigned i,
                                                         const Parameters &parameters, unsigned *indices,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,BoosterHandle booster,const char *every_points) {
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_0_1.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_ge90.csv", std::ios::out | std::ios::app);
        std::ofstream csvFile(every_points, std::ios::app);
        auto start = std::chrono::steady_clock::now();

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        float predicted_recall=0;
        int visited_points=0;
        int duration=0;
        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts==stability_times_r1){
                    int test=1;
                }

                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/K)*pre_now_stability_insert >= target_recall || first_flag || stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }

                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    current_insert_flag=true;

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=9;
                            int num_feats = feat_num;
//                            int num_feats = feat_num+ dimension_;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = static_cast<float>(visited_points)/K;
                            features[4] = duration;
                            features[5] = nn_dist;
                            features[6] = furthest_dist;
                            features[7] = mean_distance;
                            features[8] = variance_of_distances;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;
                        }

                    }
                }
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        auto end = std::chrono::steady_clock::now();
        if(k>=K){
            double search_time_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();

            float recall_raw = calculate_precision(i, retset, K);
            csvFile << i << ","<< predicted_recall << ","<< recall_raw << ","<< search_time_ms << "\n";
        }

        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::SearchWithOptGraph_pre_recall_lessPre_test(const float *query, size_t K,unsigned i,
                                                         const Parameters &parameters, unsigned *indices,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,BoosterHandle booster,const char *every_points) {
        std::ofstream csvFile("/root/deep_feature.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_ge90.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile(every_points, std::ios::app);
//        auto start = std::chrono::steady_clock::now();

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        float predicted_recall=0;
        int visited_points=0;
        int duration=0;
        int k = 0;

        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;

        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts >= stability_times){
                    int test=1;
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];


                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + i * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }
                    csvFile << i << ","
                            << step << ","
                            << distance_counts << ","
                            << inserts << ","
                            << first_nn_dist << ","
                            << nn_dist << ","
                            << furthest_dist << ","
                            << mean_distance << ","
                            << variance_of_distances << ","
                            << avg_hop << ","
                            << avg_indegree << ","
                            << density << ","
                            << entry_query_dist_ratio << ","
                            << dist_distribution_entropy << ","
                            << skewness << ","
                            << kurtosis << ","
                            << energy << ","
                            << cosine_mean << ","
                            << cosine_variance << ","
                            << cosine_direction_entropy << ","
                            << 1 << "\n";
                    break;
                }

                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/K)*pre_now_stability_insert >= target_recall || first_flag || stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }

                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    current_insert_flag=true;

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=9;
                            int num_feats = feat_num;
//                            int num_feats = feat_num+ dimension_;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = static_cast<float>(visited_points)/K;
                            features[4] = duration;
                            features[5] = nn_dist;
                            features[6] = furthest_dist;
                            features[7] = mean_distance;
                            features[8] = variance_of_distances;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;
                        }

                    }
                }
                entry_point_processed=false;
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_pre_recall_lessPre_darthFeature(const float *query, size_t K,unsigned i,
                                                         const Parameters &parameters, unsigned *indices,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,BoosterHandle booster) {
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test_0_1.csv", std::ios::out | std::ios::app);
//        std::ofstream csvFile("/root/NSG/experiment/0-k100-full/4-our_RAET/95test.csv", std::ios::out | std::ios::app);
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        int target =static_cast<int>(std::round(target_recall * 100.0f));
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        float predicted_recall=0;
        int visited_points=0;
        int duration=0;
        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }
                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/K)*pre_now_stability_insert >= target_recall || first_flag || stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }
                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            unsigned feat_num=11;
                            int num_feats = feat_num;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            double percentile_25 = retset[K*0.25].distance;
                            double percentile_50 = retset[K*0.5].distance;
                            double percentile_75 = retset[K*0.75].distance;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = first_nn_dist;
                            features[4] = nn_dist;
                            features[5] = furthest_dist;
                            features[6] = mean_distance;
                            features[7] = variance_of_distances;
                            features[8] = percentile_25;
                            features[9] = percentile_50;
                            features[10] = percentile_75;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;

//                            if(i==0 || i==1 || i==5 || i==8){
//                                float recall_raw = calculate_precision(i, retset, K);
//                                csvFile<< i << "," << distance_counts  << ","<< stability_counts  << ","<< static_cast<float>(visited_points)/K << ","<< predicted_recall  << ","<< recall_raw  <<","<<"1" << "\n";
//                            }

                        }
//                        else {
//                            if (i==0 || i==1 || i==5 || i==8) {
//                                float recall_raw = calculate_precision(i, retset, K);
//                                csvFile<< i << "," << distance_counts  << ","<< stability_counts  << ","<< static_cast<float>(visited_points)/K << ","<< predicted_recall  << ","<< recall_raw  <<","<<"0" << "\n";
//                            }
//                        }
                    }
                }
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
//        float recall_raw = calculate_precision(i, retset, K);
//        csvFile<< i << "," << distance_counts  << ","<< predicted_recall  << ","<< recall_raw  << "\n";
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_Raet_test_metrics(const float *query, size_t K,
                                                         const Parameters &parameters, unsigned *indices,unsigned i,
                                                         const float target_recall, size_t stability_times,size_t stability_times_r1,BoosterHandle booster,unsigned &success_stop_count,unsigned &stop_count,const char *train_feature_filename) {

        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        float target = std::round(target_recall * 100.0) / 100.0;
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        double first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //此次搜索中准确率为1的训练数据个数,用来限制训练集特征数据输出标签为1的行数
        int full_accuracy_count = 0;
        // 用于存储连续 Jaccard 相似度 大于 target_recall 的次数
        int stability_counts = 0;
        //模型调用次数
        int use_model_count = 0;
        //退出外层循环标志
        bool stop_flag = false;
        //搜索到非0位置标志
        bool flag_CNS_noindex0=false;
        //是否可以进行模型调用
        bool flag_predict=false;
        bool first_flag=true;
        //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
        float outer_layer_predict_recall=0;
        int pre_now_stability_insert = 0;
        int visited_points=0;
        int duration=0;

        int k = 0;
        while (k < (int) L) {
            int nk = L;
            if (retset[k].flag) {
                if(k>0){
                    flag_CNS_noindex0=true;
                }
                if(stability_counts >= stability_times && (outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target_recall || first_flag ||stability_counts==stability_times_r1)){
                    flag_predict=true;
                    first_flag=false;
                }
                unsigned visited_points_now = 0;
                for (int i = 0; i < K; i++) {
                    if (retset[i].flag == false) {
                        visited_points_now++;
                    }
                }
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                //上一次模型预测结果
                double prev_recall = 0;

                //第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre = true;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;

                    if (r < K) {
                        ++current_insert_count;
                        current_insert_flag = true;
                        ++pre_now_stability_insert;
                        prev_recall = prev_recall + 1.0 / K;
                    }
                    if(flag_predict){
                        if (prev_recall >= target_recall || first_model_pre){
                            auto start_1 = std::chrono::high_resolution_clock::now();
                            model_pre_cout++;
                            unsigned feat_num=9;
                            int num_feats = feat_num ;
                            std::vector<double> features(num_feats); // 假设特征类型为 double
                            std::vector<double> output(1); // 假设输出类型为 double
                            float nn_dist=retset[0].distance;
                            float furthest_dist=retset[K-1].distance;

                            //前K个数据点的平均距离和方差
                            double sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                sum += retset[i].distance;
                            }
                            double mean_distance = sum / K; // 平均距离
                            // 2. 计算方差（整数部分）
                            double sq_sum = 0.0;
                            for (int i = 0; i < K; ++i) {
                                double diff = retset[i].distance - mean_distance;
                                sq_sum += diff * diff; // 避免用pow()提高效率
                            }
                            double variance_of_distances = sq_sum / K;
                            features[0] = step;
                            features[1] = distance_counts;
                            features[2] = inserts;
                            features[3] = static_cast<float>(visited_points) /K;
                            features[4] = duration;
                            features[5] = nn_dist;
                            features[6] = furthest_dist;
                            features[7] = mean_distance;
                            features[8] = variance_of_distances;
//                            for (int i = 0; i < dimension_; ++i) {
//                                features[i + feat_num] = static_cast<double>(query[i]);
//                            }

                            double out_result[1];
                            long int out_len;
                            int res = LGBM_BoosterPredictForMatSingleRow(
                                    booster,
                                    features.data(),
                                    C_API_DTYPE_FLOAT64,
                                    num_feats,
                                    1,
                                    C_API_PREDICT_NORMAL,
                                    0,
                                    -1,
                                    "",
                                    &out_len,
                                    out_result);
                            float predicted_recall = (float) out_result[0];
                            predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));
                            //记录模型调用时间消耗
                            auto end_1 = std::chrono::high_resolution_clock::now();
                            time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                            if (predicted_recall >= target_recall) {
                                //真实准确率
                                float recall_raw = calculate_precision(i, retset, K);
                                if(recall_raw >= target_recall){
                                    success_stop_count++;
                                }
                                stop_count++;
                                stop_flag = true;
                                break;
                            }
                            prev_recall=predicted_recall;
                            first_model_pre=false;
                            //更新外层，记录的是两次稳定之间的插入量
                            pre_now_stability_insert=0;
                            //更新外层模型预测值为最新值
                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if(current_insert_count==0 && stability_counts==stability_times_r1){
                    stability_times_r1++;
                }
                //如果调用模型预测结果超过阈值，那么就终止该算法
                if (stop_flag) break;
                if (flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0) {
                        stability_counts++;
                    } else {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-target)，即相似数量为 K*target ，那么认为一次稳定
                        if (current_insert_count*100 <= K * (100 - target)) {
                            stability_counts++;
                        } else {
                            stability_counts = 0;
                        }
                    }
                }
            }
            flag_predict=false;
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_our_Raet_CNS_test_regerssion_classification(const float *query, size_t K,
                                                                   const Parameters &parameters, unsigned *indices, unsigned qid ,const float target,
                                                                   BoosterHandle booster_regression,BoosterHandle booster_classification) {
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
//        float target = std::round(target_recall * 100.0) / 100.0;
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;
        bool flag_model_predict=true;
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1 && flag_model_predict){
                    auto start_1 = std::chrono::high_resolution_clock::now();
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];

                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }


                    unsigned feat_num=15;
                    int num_feats = feat_num;
                    std::vector<double> features(num_feats); // 假设特征类型为 double
                    std::vector<double> output(1); // 假设输出类型为 double
                    features[0] = step;
                    features[1] = distance_counts;
                    features[2] = inserts;
                    features[3] = first_nn_dist;
                    features[4] = nn_dist;
                    features[5] = furthest_dist;
                    features[6] = mean_distance;
                    features[7] = variance_of_distances;
                    features[8] = avg_hop;
                    features[9] = avg_indegree;

                    features[10] = entry_query_dist_ratio;
                    features[11] = dist_distribution_entropy;

                    features[12] = cosine_mean;
                    features[13] = cosine_variance;
                    features[14] = cosine_direction_entropy;

//                    for (int i = 0; i < dimension_; ++i) {
//                        features[i + feat_num] = static_cast<double>(query[i]);
//                    }

                    double out_result[1];
                    long int out_len;
                    int res = LGBM_BoosterPredictForMatSingleRow(
                            booster_classification,
                            features.data(),
                            C_API_DTYPE_FLOAT64,
                            num_feats,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            -1,
                            "",
                            &out_len,
                            out_result);
                    // probability of class=1
                    double prob = out_result[0];
                    // threshold = 0.5 (you can tune it)
                    int cls = (prob >= 0.9) ? 1 : 0;
                    // map to CNS
                    int predicted_CNS = (cls == 0) ? K : L;
                    flag_model_predict=false;
                    //记录模型调用时间消耗
                    auto end_1 = std::chrono::high_resolution_clock::now();
                    time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    if(predicted_CNS==K){
                        retset.resize(predicted_CNS+1);
                        L=predicted_CNS;
                    }else{
                        double out_result[1];
                        long int out_len;
                        int res = LGBM_BoosterPredictForMatSingleRow(
                                booster_regression,
                                features.data(),
                                C_API_DTYPE_FLOAT64,
                                num_feats,
                                1,
                                C_API_PREDICT_NORMAL,
                                0,
                                -1,
                                "",
                                &out_len,
                                out_result);
                        int predicted_CNS = out_result[0];
                        predicted_CNS = std::max((int)K, predicted_CNS);
                        predicted_CNS = std::min((int)L, predicted_CNS);
                        //记录模型调用时间消耗
                        auto end_1 = std::chrono::high_resolution_clock::now();
                        time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                        retset.resize(predicted_CNS+1);
                        L=predicted_CNS;
                        flag_model_predict=false;
                    }
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
    void IndexNSG::SearchWithOptGraph_our_Raet_CNS_test_classification(const float *query, size_t K,
                                                                                  const Parameters &parameters, unsigned *indices, unsigned qid ,const float target,
                                                                                  BoosterHandle booster_classification) {
        unsigned step=0;
        unsigned distance_counts=0;
        unsigned inserts=0;
//        float target = std::round(target_recall * 100.0) / 100.0;
        unsigned L = parameters.Get<unsigned>("L_search");
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        // std::mt19937 rng(rand());
        // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;
        unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
        unsigned MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id]) continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_) continue;
            float *x = (float *) (opt_graph_ + node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
            distance_counts++;
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }
        // std::cout<<L<<std::endl;

        std::sort(retset.begin(), retset.begin() + L);
        float first_nn_dist=retset[0].distance;
        //此次查询中，有多少数据点加入了候选邻居集合中前k位置
        double insert_K_count = 0;
        //初始化跳数
        unsigned hop_c=0;
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        bool entry_point_processed = true;
        float entry_dist = retset[0].distance;
        double entry_neighbors_sum = 0.0;
        int entry_neighbors_cnt = 0;
        bool flag_model_predict=true;
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned *neighbors = (unsigned *) (opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                if(entry_point_processed){
                    entry_neighbors_cnt=MaxM;
                }
                hop_c+=1;
                hop_Count[n]=hop_c;

                //记录当前数据点有多少邻居加入了CNS中前K位置
                unsigned current_insert_count = 0;

                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    hop_Count[id]=hop_c+1;
                    if (flags[id]) continue;
                    flags[id] = 1;

                    //判断当前邻居，有没有插入到前K位置
                    bool current_insert_flag = false;

                    float *data = (float *) (opt_graph_ + node_size * id);
                    float norm = *data;
                    data++;
                    float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
                    distance_counts++;
                    if(entry_point_processed){
                        entry_neighbors_sum+=dist;
                    }
                    if (dist >= retset[L - 1].distance) continue;
                    inserts++;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    // if(L+1 < retset.size()) ++L;
                    if (r < nk) nk = r;
                    if (r < K) {
                        ++insert_K_count;
                        ++current_insert_count;
                        current_insert_flag = true;
                    }
                }
                entry_point_processed=false;
                //收集特征
                if(k==K-1 && flag_model_predict){
                    auto start_1 = std::chrono::high_resolution_clock::now();
                    //收集特征
                    float nn_dist=retset[0].distance;
                    float nn10_dist=retset[9].distance;
                    float nn_to_first = nn_dist/first_nn_dist;
                    float nn10_to_first = nn10_dist/first_nn_dist;
                    float furthest_dist=retset[K-1].distance;

                    //距离分布熵
                    double dist_distribution_entropy = 0.0;

                    constexpr int DIST_B = 20;
                    float hist[DIST_B] = {0};

                    float d_min = retset[0].distance;

                    // 1️⃣ 计算 max_delta
                    float max_delta = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        float delta = retset[i].distance - d_min;
                        if (delta > max_delta)
                            max_delta = delta;
                    }
                    // 2️⃣ 若所有距离相同 → 熵 = 0
                    if (max_delta < 1e-12f) {
                        dist_distribution_entropy = 0.0;
                    } else {
                        // 3️⃣ 正常分桶
                        for (int i = 0; i < K; ++i) {
                            float delta = retset[i].distance - d_min;   // >= 0
                            float u = delta / max_delta;                // ∈ [0,1]
                            int b = int(u * DIST_B);
                            if (b >= DIST_B) b = DIST_B - 1;            // clamp
                            hist[b] += 1.0f;
                        }

                        // 4️⃣ 归一化 + 熵
                        for (int i = 0; i < DIST_B; ++i) {
                            float p = hist[i] / K;
                            if (p > 0)
                                dist_distribution_entropy -= p * std::log(p);
                        }
                    }

                    double avg_indegree=0.0;
                    double avg_hop=0.0;
                    //前K个数据点的平均距离和方差
                    double sum = 0.0;

                    int valid=0;
                    //energy
                    float energy=0;
                    //skewness计算
                    float sum_of_cubes = 0;
                    float sum_of_squares = 0;
                    for (int i = 0; i < K; ++i) {
                        sum += retset[i].distance;
                        avg_indegree+=indegree[retset[i].id];
                        avg_hop+=hop_Count[retset[i].id];

                        sum_of_squares += retset[i].distance * retset[i].distance;
                        sum_of_cubes += retset[i].distance * retset[i].distance * retset[i].distance;
                    }
                    double mean_distance = sum / K; // 平均距离
                    avg_hop /=K;
                    avg_indegree /= K;

                    // 2. 计算方差（整数部分）
                    double sq_sum = 0.0;
                    for (int i = 0; i < K; ++i) {
                        double diff = retset[i].distance - mean_distance;
                        sq_sum += diff * diff; // 避免用pow()提高效率
                    }
                    double variance_of_distances = sq_sum / K;
                    float variance = (sum_of_squares / K) - (mean_distance * mean_distance);
                    float skewness = (sum_of_cubes / K) - (3 * mean_distance * variance) - (mean_distance * mean_distance * mean_distance);

                    double density = 1.0/mean_distance;
                    double entry_query_dist_ratio = entry_dist * entry_neighbors_cnt / entry_neighbors_sum;
                    //energy计算
                    energy=sum_of_squares;
                    //kurtosis计算
                    float m2 = 0;  // Second moment (variance)
                    float m4 = 0;  // Fourth moment
                    for (int i = 0; i < K; i++) {
                        float dist = retset[i].distance;
                        float diff = dist - mean_distance;
                        m2 += diff * diff;
                        m4 += diff * diff * diff * diff;
                    }
                    m2 /= K;  // Variance
                    m4 /= K;
                    float kurtosis = (m4 / (m2 * m2)) - 3;

                    //余弦相似度计算
                    const float* query = norm_queries + qid * dimension_;
                    std::vector<float> sims;
                    sims.reserve(K);
                    // 1️⃣ 直接计算余弦（实际上是 dot）
                    // 2️⃣ 均值
                    float cosine_mean = 0.0f;
                    for (int i = 0; i < K; ++i) {
                        unsigned id = retset[i].id;
                        const float* vec = norm_db + id * dimension_;
                        float sim = 0.0f;
                        for (size_t i = 0; i < dimension_; ++i) {
                            sim += query[i] * vec[i];
                        }
                        cosine_mean += sim;
                        sims.push_back(sim);
                    }
                    cosine_mean /= sims.size();
                    // 3️⃣ 方差
                    float cosine_variance = 0.0f;
                    for (float s : sims) cosine_variance += (s - cosine_mean) * (s - cosine_mean);
                    cosine_variance /= sims.size();
                    // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
                    const int B = 20;
                    std::vector<float> hist_c(B, 0.0f);
                    for (float s : sims) {
                        // s ∈ [-1,1] 线性映射到 [0,B)
                        float u = (s + 1.0f) * 0.5f;
                        int b = std::min(B - 1, int(u * B));
                        hist_c[b] += 1.0f;
                    }
                    for (float &h : hist_c) h /= sims.size();
                    float cosine_direction_entropy = 0.0f;
                    for (float h : hist_c){
                        if (h > 0)
                            cosine_direction_entropy -= h * std::log(h);
                    }


                    unsigned feat_num=15;
                    int num_feats = feat_num;
                    std::vector<double> features(num_feats); // 假设特征类型为 double
                    std::vector<double> output(1); // 假设输出类型为 double
                    features[0] = step;
                    features[1] = distance_counts;
                    features[2] = inserts;
                    features[3] = first_nn_dist;
                    features[4] = nn_dist;
                    features[5] = furthest_dist;
                    features[6] = mean_distance;
                    features[7] = variance_of_distances;
                    features[8] = avg_hop;
                    features[9] = avg_indegree;

                    features[10] = entry_query_dist_ratio;
                    features[11] = dist_distribution_entropy;

                    features[12] = cosine_mean;
                    features[13] = cosine_variance;
                    features[14] = cosine_direction_entropy;

//                    for (int i = 0; i < dimension_; ++i) {
//                        features[i + feat_num] = static_cast<double>(query[i]);
//                    }

                    double out_result[1];
                    long int out_len;
                    int res = LGBM_BoosterPredictForMatSingleRow(
                            booster_classification,
                            features.data(),
                            C_API_DTYPE_FLOAT64,
                            num_feats,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            -1,
                            "",
                            &out_len,
                            out_result);
                    // probability of class=1
                    double prob = out_result[0];
                    // threshold = 0.5 (you can tune it)
                    int cls = (prob >= 0.9) ? 1 : 0;
                    // map to CNS
                    int predicted_CNS = (cls == 0) ? K : L;
                    flag_model_predict=false;
                    //记录模型调用时间消耗
                    auto end_1 = std::chrono::high_resolution_clock::now();
                    time_spend_model += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1);
                    if(predicted_CNS==K){
                        retset.resize(predicted_CNS+1);
                        L=predicted_CNS;
                    }
                }
            }
            step++;
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

        data_ = data;
        data_len = (dimension_ + 1) * sizeof(float);
        neighbor_len = (width + 1) * sizeof(unsigned);
        node_size = data_len + neighbor_len;
        opt_graph_ = (char *) malloc(node_size * nd_);
        DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
        for (unsigned i = 0; i < nd_; i++) {
            char *cur_node_offset = opt_graph_ + i * node_size;
            float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
            std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
            std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                        data_len - sizeof(float));

            cur_node_offset += data_len;
            unsigned k = final_graph_[i].size();
            std::memcpy(cur_node_offset, &k, sizeof(unsigned));
            std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                        k * sizeof(unsigned));
            std::vector<unsigned>().swap(final_graph_[i]);
        }
        CompactGraph().swap(final_graph_);
    }

    void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = nd_ + 1;
            for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
                if (flag[final_graph_[tmp][i]] == false) {
                    next = final_graph_[tmp][i];
                    break;
                }
            }
            // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
            if (next == (nd_ + 1)) {
                s.pop();
                if (s.empty()) break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }

    void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                            const Parameters &parameter) {
        unsigned id = nd_;
        for (unsigned i = 0; i < nd_; i++) {
            if (flag[i] == false) {
                id = i;
                break;
            }
        }

        if (id == nd_) return;  // No Unlinked Node

        std::vector<Neighbor> tmp, pool;
        get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
        std::sort(pool.begin(), pool.end());

        unsigned found = 0;
        for (unsigned i = 0; i < pool.size(); i++) {
            if (flag[pool[i].id]) {
                // std::cout << pool[i].id << '\n';
                root = pool[i].id;
                found = 1;
                break;
            }
        }
        if (found == 0) {
            while (true) {
                unsigned rid = rand() % nd_;
                if (flag[rid]) {
                    root = rid;
                    break;
                }
            }
        }
        final_graph_[root].push_back(id);
    }

    void IndexNSG::tree_grow(const Parameters &parameter) {
        unsigned root = ep_;
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < nd_) {
            DFS(flags, root, unlinked_cnt);
            // std::cout << unlinked_cnt << '\n';
            if (unlinked_cnt >= nd_) break;
            findroot(flags, root, parameter);
            // std::cout << "new root"<<":"<<root << '\n';
        }
        for (size_t i = 0; i < nd_; ++i) {
            if (final_graph_[i].size() > width) {
                width = final_graph_[i].size();
            }
        }
    }
}
