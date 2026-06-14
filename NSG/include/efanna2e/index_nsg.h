#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <LightGBM/c_api.h>
#include <LightGBM/boosting.h>
#include <LightGBM/config.h>
#include <LightGBM/prediction_early_stop.h>

namespace efanna2e {

    class IndexNSG : public Index {
    public:
        explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


        virtual ~IndexNSG();

        virtual void Save(const char *filename) override;

        virtual void Load(const char *filename) override;


        virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

        virtual void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) override;

        void SearchWithOptGraph(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices);
        void SearchWithOptGraph_get_every_points_recall(
                const float *query,
                size_t K,
                unsigned i,
                const Parameters &parameters,
                unsigned *indices,const char *train_feature_filename);

        void SearchWithOptGraph_Raet_CNS_train_data(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices, const float acc_thold, unsigned i, const char *train_feature_filename);

        void SearchWithOptGraph_Laet_Darth_train_data(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices, unsigned i, const char *train_feature_filename);

        void SearchWithOptGraph_laet_test(
                const float *query,
                size_t K,
                unsigned i,
                const Parameters &parameters,
                unsigned *indices,
                unsigned laet_F,
                float laet_prediction_multiplier, BoosterHandle booster,const char *train_feature_filename);

        void SearchWithOptGraph_Darth_test_time(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices,
                unsigned i,
                unsigned ipi,
                unsigned mpi,
                float target_recall,
                BoosterHandle booster,
                const char *train_feature_filename);

        void SearchWithOptGraph_Darth_test_metrics(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices,
                unsigned i,
                unsigned ipi,
                unsigned mpi,
                float target_recall,
                BoosterHandle booster,
                unsigned &success_stop_count,
                unsigned &stop_count);

        void SearchWithOptGraph_stablize(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices, const float acc_thold, unsigned i, int& all_stability_counts,const char *stablity_filename);
        void SearchWithOptGraph_Raet_recall_train_data(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices, unsigned i, const char *train_feature_filename);

        void OptimizeGraph(float *data);

        void SearchWithOptGraph_pre_recall(const float *query, size_t K,
                                           const Parameters &parameters,
                                           unsigned *indices, const float acc_thold, size_t stability_times,BoosterHandle booster);

        void SearchWithOptGraph_pre_recall_lessPre(const float *query, size_t K,unsigned i,
                                                   const Parameters &parameters,
                                                   unsigned *indices, const float acc_thold, size_t stability_times,size_t stability_times_r1,
                                                   BoosterHandle booster,const char *train_feature_filename);
        void SearchWithOptGraph_pre_recall_seletor_label(const float *query, size_t K,unsigned i,
                                                   const Parameters &parameters,
                                                   unsigned *indices, const float acc_thold, size_t stability_times,size_t stability_times_r1,
                                                   BoosterHandle booster,const char *train_feature_filename);
        void SearchWithOptGraph_pre_recall_lessPre_test(const float *query, size_t K,unsigned i,
                                                   const Parameters &parameters,
                                                   unsigned *indices, const float acc_thold, size_t stability_times,size_t stability_times_r1,
                                                   BoosterHandle booster,const char *train_feature_filename);
        void SearchWithOptGraph_pre_recall_lessPre_darthFeature(const float *query, size_t K,unsigned i,
                                                   const Parameters &parameters,
                                                   unsigned *indices, const float acc_thold, size_t stability_times,size_t stability_times_r1,
                                                   BoosterHandle booster);
        void SearchWithOptGraph_Raet_test_metrics(
                const float *query,
                size_t K,
                const Parameters &parameters,
                unsigned *indices,
                unsigned i,
                const float target_recall,
                size_t stability_times,
                size_t stability_times_r1,
                BoosterHandle booster,
                unsigned &success_stop_count,
                unsigned &stop_count,
                const char *train_feature_filename);
        void SearchWithOptGraph_our_Raet_CNS_test_regression(const float *query, size_t K,
                                                   const Parameters &parameters,
                                                   unsigned *indices, const float acc_thold,unsigned qid,
                                                   BoosterHandle booster);
        void SearchWithOptGraph_our_Raet_CNS_test_regression_Noquery_Noske_Nodensty(const float *query, size_t K,
                                                             const Parameters &parameters,
                                                             unsigned *indices, const float acc_thold,unsigned qid,
                                                             BoosterHandle booster,const char *train_feature_filename);
        void SearchWithOptGraph_Raet_test_regression_metrics(const float *query, size_t K, const Parameters &parameters,
                                                             unsigned int *indices, unsigned int qid,
                                                             const float target_recall,
                                                             BoosterHandle booster, unsigned int &success_stop_count);
        void SearchWithOptGraph_our_Raet_CNS_test_regerssion_classification(const float *query, size_t K,
                                                             const Parameters &parameters,
                                                             unsigned *indices, unsigned qid,const float target_recall,
                                                             BoosterHandle booster_regerssion,BoosterHandle booster_classification);
        void SearchWithOptGraph_our_Raet_CNS_test_classification(const float *query, size_t K,
                                                                            const Parameters &parameters,
                                                                            unsigned *indices, unsigned qid,const float target_recall,
                                                                            BoosterHandle booster_classification);
        void SearchWithOptGraph_our_Raet_CNS_selector(const float *query, size_t K,
                                                                            const Parameters &parameters,
                                                                            unsigned *indices, unsigned qid,const float target_recall,size_t stability_times,size_t stability_times_r1,
                                                                            BoosterHandle booster_recall,BoosterHandle booster_CNS,const char *train_feature_filename);

        //模型预测时间统计消耗
        void InitTimeSpendModel() { time_spend_model = std::chrono::nanoseconds::zero(); }

        size_t GetTimeSpendModel() { return time_spend_model.count(); }

        //距离计算次数
        void InitDistCount() { dist_cout = 0; }

        size_t GetDistCount() { return dist_cout; }

        //模型预测次数
        void InitModelPreCount() { model_pre_cout = 0; }

        size_t GetModelPreCount() { return model_pre_cout; }
        //获取每个数据点的入度值
        void GetIndegree();
        //归一化
        void Normalization_db_and_query(float *vecsDB,float *vecsQ,unsigned n,unsigned nQ,unsigned d){
            norm_db = new float[(size_t)n * d];
            norm_queries = new float[(size_t)nQ * d];
            // ----- 拷贝原始 DB -----
            memcpy(norm_db, vecsDB, sizeof(float) * n * d);
            // ----- 拷贝查询向量 -----
            memcpy(norm_queries, vecsQ, sizeof(float) * nQ * d);
            // ===== ⭐ 2. 归一化函数 =====
            auto normalize = [&](float* vec, size_t nvec, size_t dim){
                for(size_t i = 0; i < nvec; i++){
                    float norm = 0.0f;
                    float* v = vec + i * dim;

                    for(size_t j = 0; j < dim; j++){
                        norm += v[j] * v[j];
                    }

                    norm = std::sqrt(norm);

                    if(norm > 0){
                        float inv = 1.0f / norm;
                        for(size_t j = 0; j < dim; j++){
                            v[j] *= inv;
                        }
                    }
                }
            };
            // ===== ⭐ 3. 真正归一化（不影响原始向量）=====
            normalize(norm_db, n, d);
            normalize(norm_queries, nQ, d);
        }

        //加载laet预测准确率模型
        bool Predict_laet_LoadModel(const std::string &file) {
            // 释放旧模型（如果存在）
            laet_model.reset();
            // 加载新模型
            laet_model.reset(LightGBM::Boosting::CreateBoosting(std::string("gbdt"), file.c_str()));
            if (!laet_model) {
                std::cerr << "Failed to load LightGBM model: " << file << std::endl;
                return false;
            }
            return true;
        };

        //加载预测准确率模型
        bool Predict_recall_LoadModel_Regression(const std::string &file) {
            // 释放旧模型（如果存在）
            booster_regression.reset();
            // 加载新模型
            booster_regression.reset(LightGBM::Boosting::CreateBoosting(std::string("gbdt"), file.c_str()));
            if (!booster_regression) {
                std::cerr << "Failed to load LightGBM model: " << file << std::endl;
                return false;
            }
            return true;
        };

        void load_ivecs_data(const char *filename, unsigned &num, unsigned &dim, unsigned query_num) {
            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "open file error" << std::endl;
                exit(-1);
            }
            // 1. 读取向量维度
            in.read((char *) &dim, 4);
            // 2. 计算文件中真实向量数
            in.seekg(0, std::ios::end);
            size_t fsize = (size_t) in.tellg();
            unsigned total_num = (unsigned) (fsize / ((dim + 1) * 4));
            // 3. 实际读取数量
            num = std::min(query_num, total_num);
            // 4. 分配空间（只分配需要的）
            true_data.clear();
            true_data.resize(num);
            for (unsigned i = 0; i < num; i++) {
                true_data[i].resize(dim);
            }
            // 5. 回到文件开头
            in.seekg(0, std::ios::beg);
            // 6. 只读取 num 条
            for (unsigned i = 0; i < num; i++) {
                in.seekg(4, std::ios::cur);                 // 跳过 dim
                in.read((char *) true_data[i].data(), dim * 4);
            }
            in.close();
        }

        //计算准确率
        float calculate_precision(unsigned query_index, std::vector<Neighbor> &retset, int top_k) {
            float acc = 0;
            for (size_t j = 0; j < top_k; j++) {
                for (size_t m = 0; m < top_k; m++) {
                    if (retset[j].id == true_data[query_index][m]) {
                        acc++;
                        break;
                    }
                }
            }
            return acc / top_k;
        }

    protected:
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<SimpleNeighbors> LockGraph;
        typedef std::vector<nhood> KNNGraph;

        CompactGraph final_graph_;

        Index *initializer_ = nullptr;

        void init_graph(const Parameters &parameters);

        void get_neighbors(
                const float *query,
                const Parameters &parameter,
                std::vector<Neighbor> &retset,
                std::vector<Neighbor> &fullset);

        void get_neighbors(
                const float *query,
                const Parameters &parameter,
                boost::dynamic_bitset<> &flags,
                std::vector<Neighbor> &retset,
                std::vector<Neighbor> &fullset);

        //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);

        void
        sync_prune(unsigned q, std::vector<Neighbor> &pool, const Parameters &parameter, boost::dynamic_bitset<> &flags,
                   SimpleNeighbor *cut_graph_);

        void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);

        void Load_nn_graph(const char *filename);

        void tree_grow(const Parameters &parameter);

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


    private:
        unsigned width;
        unsigned ep_;
        std::vector<std::mutex> locks;
        char *opt_graph_ = nullptr;
        size_t node_size;
        size_t data_len;
        size_t neighbor_len;
        KNNGraph nnd_graph;
        //groundtruth
        std::vector<std::vector<unsigned>> true_data;
        //模型预测花费时间
        std::chrono::nanoseconds time_spend_model;
        //距离计算次数
        size_t dist_cout;
        //模型预测次数
        size_t model_pre_cout;
        //存储每个数据点，从入口点开始所经过的跳数
        std::vector<int> hop_Count;
        //存储数据点的入度
        std::vector<int> indegree;
        //余弦相似度相关，归一化
        float* norm_db;
        float* norm_queries;
        //laet模型
        std::unique_ptr<LightGBM::Boosting> laet_model;
        //回归模型预测准确率
        std::unique_ptr<LightGBM::Boosting> booster_regression;
        LightGBM::PredictionEarlyStopInstance early_stop;
        LightGBM::PredictionEarlyStopConfig tree_config;
        LightGBM::PredictionEarlyStopInstance tree_early_stop =
                LightGBM::CreatePredictionEarlyStopInstance(std::string("none"), tree_config);
    };
}

#endif //EFANNA2E_INDEX_NSG_H
