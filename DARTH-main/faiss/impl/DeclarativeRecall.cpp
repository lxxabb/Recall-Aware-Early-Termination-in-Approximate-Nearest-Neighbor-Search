#include "DeclarativeRecall.h"
#include "/home/extra_home/lxx23125236/ali/DARTH-main/hnsw-test/ComUtil.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>


namespace faiss {

// DeclarativeRecallDataManager
    DeclarativeRecallDataManager::DeclarativeRecallDataManager() {}

    DeclarativeRecallDataManager::DeclarativeRecallDataManager(
            float *distances,
            idx_t *labels,
            idx_t *gt,
            float *gt_dist,
            idx_t nq,
            idx_t d,
            idx_t k,
            const float *queries,
            char *log_filename,
            const float *db,
            idx_t ndb,
            idx_t *gt_for_all_k,
            idx_t k_all,
            const std::vector<int> *out_degree_L0,
            std::vector<int> *node_hop,
            std::vector<int> *node_hop_qid,
            const float* norm_queries,
            const float* norm_db)
            : distances(distances),
              labels(labels),
              gt(gt),
              gt_dist(gt_dist),
              nq(nq),
              ndb(ndb),
              k(k),
              d(d),
              queries(queries),
              log_filename(log_filename),
              db(db),
              gt_for_all_k(gt_for_all_k),
              k_all(k_all),
              out_degree_L0(out_degree_L0),
              node_hop(node_hop),
              node_hop_qid(node_hop_qid),
            // ⭐⭐ 初始化
              norm_queries(norm_queries),
              norm_db(norm_db){
//        // ===== 新增 =====,计算两次模型预测之间的CNS中不同数据点数量
//        last_pred_mark.resize(ndb, -1);
//        pred_epoch = 0;
    }

    FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN

    int DeclarativeRecallDataManager::get_out_degree(idx_t node) const {
        if (!out_degree_L0) return -1;
        return (*out_degree_L0)[node];
    }

    void DeclarativeRecallDataManager::record_hop(idx_t node, int hop, idx_t q_id) {
        if ((*node_hop_qid)[node] != q_id) {
            (*node_hop_qid)[node] = q_id;
            (*node_hop)[node] = hop;
        }
    }
//    //在last_pred_mark中记录上一次模型预测CNS中数据点
//    void DeclarativeRecallDataManager::snapshot_labels(idx_t q_id) {
//        pred_epoch++;
//        idx_t base = q_id * k;
//        for (idx_t i = 0; i < k; i++) {
//            idx_t v = labels[base + i];
//            if (v >= 0) {
//                last_pred_mark[v] = pred_epoch;
//            }
//        }
//    }
//    //计算上一次模型预测和当前模型预测的CNS中不一样的数据点个数
//    int DeclarativeRecallDataManager::count_new_labels_since_last_snapshot(
//            idx_t q_id) const {
//        int cnt = 0;
//        idx_t base = q_id * k;
//
//        for (idx_t i = 0; i < k; i++) {
//            idx_t v = labels[base + i];
//            if (v >= 0 && last_pred_mark[v] != pred_epoch) {
//                cnt++;
//            }
//        }
//        return cnt;
//    }

    float DeclarativeRecallDataManager::get_ed(const float *x, const float *y, size_t d) {
        size_t i;
        float res = 0;
        FAISS_PRAGMA_IMPRECISE_LOOP
        for (i = 0; i < d; i++) {
            const float tmp = x[i] - y[i];
            res += tmp * tmp;
        }
        return res;
    }
//    float DeclarativeRecallDataManager::cosine_similarity(
//            const float *x,
//            const float *y,
//            size_t d) {
//
//        float dot = 0.0f;
//        float nx = 0.0f;
//        float ny = 0.0f;
//
//        FAISS_PRAGMA_IMPRECISE_LOOP
//        for (size_t i = 0; i < d; ++i) {
//            float xi = x[i];
//            float yi = y[i];
//            dot += xi * yi;
//            nx  += xi * xi;
//            ny  += yi * yi;
//        }
//
//        float norm = std::sqrt(nx) * std::sqrt(ny);
//        if (norm == 0.0f) return 0.0f;
//
//        return dot / norm;   // ∈ [-1, 1]
//    }
    float DeclarativeRecallDataManager::cosine_similarity(
            const float *x,
            const float *y,
            size_t d) {
        float dot = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            dot += x[i] * y[i];
        }
        return dot;   // 现在 dot 就是 cos ∈ [-1,1]
    }



    FAISS_PRAGMA_IMPRECISE_FUNCTION_END

    double DeclarativeRecallDataManager::elapsed_secs() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    float DeclarativeRecallDataManager::get_avg_dist_of_query(idx_t q_id) {
        float sum = 0.0f;
        int cnt = 0;
        for (idx_t i = 0; i < k; i++) {
            float d = distances[q_id * k + i];
            if (d == 0.0f) continue;   // 忽略 self-match
            if (std::isnan(d)) continue;
            sum += d;
            cnt++;
        }
//        if (cnt == 0) return std::numeric_limits<float>::quiet_NaN();
        return sum / cnt;
    }

    float DeclarativeRecallDataManager::get_avg_degree_of_query(idx_t q_id) {
        int sum = 0;
        for (idx_t i = 0; i < k; i++) {
            idx_t v = labels[q_id * k + i];
            sum += (*out_degree_L0)[v];
        }
        return sum / k;
    }

    float DeclarativeRecallDataManager::get_avg_hop_of_query(idx_t q_id) {
        int sum = 0;
        for (idx_t i = 0; i < k; i++) {
            idx_t v = labels[q_id * k + i];
            sum += (*node_hop)[v];
        }
        return sum / k;
    }
    //距离分布熵：CNS中前k位置到查询点距离，与最近数据点到查询点距离比值的和
    float DeclarativeRecallDataManager::get_dist_distribution_entropy(idx_t q_id,float d0) {
//        float sum = 0;
//        int valid = 0;
//        for (idx_t i = 0; i < k; i++) {
//            float d = distances[q_id * k + i];
//            idx_t id = labels[q_id * k + i];
//            // 忽略自身
//            if (id == q_id) continue;
//            sum += d / d0;
//            valid++;
//        }
//        return sum / valid;
        if (d0 < 1e-12) return 0.0f;

        const int B = 20;
        std::vector<float> hist(B, 0.0f);
        int valid = 0;

        // 设一个“感兴趣的最大距离倍数”
        const float r_max = 10.0f;   // 经验值，和 cosine 的 [-1,1] 类似

        for (idx_t i = 0; i < k; ++i) {
            idx_t id = labels[q_id * k + i];
            if (id == q_id) continue;

            float d = distances[q_id * k + i];
            float r = d / d0;        // r >= 0

            // ===== 核心：映射到 [0,1] =====
            float u = std::min(r / r_max, 1.0f);   // u ∈ [0,1]

            int b = std::min(B - 1, int(u * B));
            hist[b] += 1.0f;
            valid++;
        }

        if (valid == 0) return 0.0f;

        // 概率化
        for (float &h : hist) h /= valid;

        // Shannon entropy
        float entropy = 0.0f;
        for (float h : hist) {
            if (h > 0.0f)
                entropy -= h * std::log(h);
        }

        return entropy;
    }

    float DeclarativeRecallDataManager::get_nearest_dist_of_query(idx_t q_id) {
        float min_dist = std::numeric_limits<float>::max();
        for (idx_t i = 0; i < k; i++) {

            float d = distances[q_id * k + i];
            idx_t id = labels[q_id * k + i];

            // 跳过自身
            if (id == q_id) continue;

            if (d < min_dist)
                min_dist = d;
        }
        return min_dist;
    }
    int DeclarativeRecallDataManager::get_visited_points_k_query(idx_t q_id,VisitedTable &vt) {
        int visited_points_k=0;
        for (idx_t i = 0; i < k; i++) {
            idx_t id = labels[q_id * k + i];
            bool vget = vt.get(id);
            visited_points_k += vget ? 1 : 0;
        }
        return visited_points_k;
    }

    idx_t DeclarativeRecallDataManager::get_nearest_id_of_query(idx_t q_id) {
        float min_dist = std::numeric_limits<float>::max();
        idx_t id = 0;
        for (idx_t i = 0; i < k; i++) {
            if (distances[q_id * k + i] < min_dist) {
                min_dist = distances[q_id * k + i];
                id = labels[q_id * k + i];
            }

        }
        return id;
    }

    float DeclarativeRecallDataManager::get_furthest_dist_of_query(idx_t q_id) {
        float max_dist = std::numeric_limits<float>::min();
        for (idx_t i = 0; i < k; i++) {
            if (distances[q_id * k + i] > max_dist)
                max_dist = distances[q_id * k + i];
        }
        return max_dist;
    }
    //将CNS中每个数据点在各个维度上的值求平均，作为新的数据点，计算该数据点与查询点q的距离
    float DeclarativeRecallDataManager::get_dist_of_query_to_medoid(idx_t q_id) {
        std::vector<float> medoid(static_cast<size_t>(d), 0.0f);

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                medoid[static_cast<size_t>(j)] += db[labels[q_id * k + i] * d + j];
            }
        }

        for (int i = 0; i < d; i++) {
            medoid[static_cast<size_t>(i)] /= k;
        }

        const float *query = queries + q_id * d;

        float ed_dist = get_ed(query, medoid.data(), d);

        return ed_dist;
    }

    //计算CNS中各个数据点与查询点q的余弦距离，包括余弦的均值、方差和熵
    CosineStats DeclarativeRecallDataManager::get_cosine_stats_of_CNS(idx_t q_id) {
        const float* query = norm_queries + q_id * d;
        std::vector<float> sims;
        sims.reserve(k);
        // 1️⃣ 直接计算余弦（实际上是 dot）
        // 2️⃣ 均值
        float mean = 0.0f;
        for (int i = 0; i < k; ++i) {
            idx_t id = labels[q_id * k + i];
            const float* vec = norm_db + id * d;
            float sim = cosine_similarity(query, vec, d);
            mean += sim;
            sims.push_back(sim);
        }
        mean /= sims.size();
        // 3️⃣ 方差
        float var = 0.0f;
        for (float s : sims) var += (s - mean) * (s - mean);
        var /= sims.size();
        // 4️⃣ 方向熵 ====== 取消 acos, 直接对 cos 分桶 ======
        const int B = 20;
        std::vector<float> hist(B, 0.0f);
        for (float s : sims) {
            // s ∈ [-1,1] 线性映射到 [0,B)
            float u = (s + 1.0f) * 0.5f;
            int b = std::min(B - 1, int(u * B));
            hist[b] += 1.0f;
        }
        for (float &h : hist) h /= sims.size();
        float entropy = 0.0f;
        for (float h : hist)
            if (h > 0)
                entropy -= h * std::log(h);
        return CosineStats{mean, var, entropy};
    }

//    CosineStats DeclarativeRecallDataManager::get_cosine_stats_of_CNS(idx_t q_id) {
//        const float* query = queries + q_id * d;
//        std::vector<float> sims;
//        sims.reserve(k);
//
//        // 1️⃣ 计算 k 个点的余弦相似度
//        for (int i = 0; i < k; ++i) {
//            idx_t id = labels[q_id * k + i];
//            const float* vec = db + id * d;
//
//            float sim = cosine_similarity(query, vec, d);
//            sims.push_back(sim);
//        }
//
//        // 2️⃣ 均值
//        float mean = 0.0f;
//        for (float s : sims) mean += s;
//        mean /= sims.size();
//
//        // 3️⃣ 方差
//        float var = 0.0f;
//        for (float s : sims) var += (s - mean) * (s - mean);
//        var /= sims.size();
//
//        // 4️⃣ 方向熵
//        const int B = 20;
//        std::vector<float> hist(B, 0.0f);
//
//        for (float s : sims) {
//            float theta = std::acos(std::max(-1.0f, std::min(1.0f, s)));
//            int b = std::min(B - 1, (int)(theta / M_PI * B));
//            hist[b] += 1.0f;
//        }
//
//        for (float &h : hist) h /= sims.size();
//
//        float entropy = 0.0f;
//        for (float h : hist)
//            if (h > 0) entropy -= h * std::log(h);
//
//        return CosineStats{mean, var, entropy};
//    }



    float DeclarativeRecallDataManager::get_recallk(idx_t query_idx) {
        std::unordered_set<idx_t> gt_set(
                gt + query_idx * k, gt + (query_idx + 1) * k);
        int matches = 0;

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(labels[query_idx * k + j])) {
                matches++;
            }
        }

        float recall = (float) matches / (float) k;
        return recall;
    }

    float DeclarativeRecallDataManager::get_kth_nearest_dist_of_query(idx_t q_id, int kth) {
        float *query_distances = distances + q_id * k;

        std::vector<float> query_distances_cp(query_distances, query_distances + k);

        std::nth_element(
                query_distances_cp.begin(),
                query_distances_cp.begin() + kth,
                query_distances_cp.end());

        float kth_nearest_dist = query_distances_cp[static_cast<size_t>(kth)];

        return kth_nearest_dist;
    }

    float DeclarativeRecallDataManager::get_variance_of_query(idx_t q_id) {
        float sum = 0;
        float sum_of_squares = 0;

        for (idx_t i = 0; i < k; i++) {
            float dist = distances[q_id * k + i];
            sum += dist;
            sum_of_squares += dist * dist;
        }

        float mean = sum / k;

        return (sum_of_squares / k) - (mean * mean);
    }

    float DeclarativeRecallDataManager::get_percentile_of_query(idx_t q_id, float percentile) {
        std::vector<float> query_distances(static_cast<size_t>(k));

        for (idx_t i = 0; i < k; i++) {
            query_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
        }

        idx_t idx = static_cast<idx_t>(percentile * (k - 1));

        std::nth_element(
                query_distances.begin(),
                query_distances.begin() + idx,
                query_distances.end());

        return query_distances[static_cast<size_t>(idx)];
    }
    //Skewness（偏度）它衡量分布是否有长尾
    float DeclarativeRecallDataManager::get_skewness_of_query(idx_t q_id) {
        float sum = 0;
        float sum_of_cubes = 0;
        float sum_of_squares = 0;

        for (idx_t i = 0; i < k; i++) {
            float dist = distances[q_id * k + i];
            sum += dist;
            sum_of_squares += dist * dist;
            sum_of_cubes += dist * dist * dist;
        }

        float mean = sum / k;
        float variance = (sum_of_squares / k) - (mean * mean);
        float skewness = (sum_of_cubes / k) - (3 * mean * variance) - (mean * mean * mean);

        return skewness;
    }
    //平方和：整体距离规模大小
    float DeclarativeRecallDataManager::get_energy_of_query(idx_t q_id) {
        float sum = 0;

        for (idx_t i = 0; i < k; i++) {
            float dist = distances[q_id * k + i];
            sum += dist * dist;
        }

        return sum;
    }
    //Kurtosis（峰度）衡量尾部厚度 + 分布尖锐程度
    float DeclarativeRecallDataManager::get_kurtosis_of_query(idx_t q_id) {
        float mean = 0;
        float m2 = 0;  // Second moment (variance)
        float m4 = 0;  // Fourth moment
        idx_t n = k;

        // Calculate the mean
        for (idx_t i = 0; i < n; i++) {
            mean += distances[q_id * k + i];
        }
        mean /= n;

        // Calculate second and fourth moments
        for (idx_t i = 0; i < n; i++) {
            float dist = distances[q_id * k + i];
            float diff = dist - mean;
            m2 += diff * diff;
            m4 += diff * diff * diff * diff;
        }

        m2 /= n;  // Variance
        m4 /= n;  // Fourth moment

        // Calculate kurtosis
        float kurtosis = (m4 / (m2 * m2)) - 3;

        return kurtosis;
    }

    float DeclarativeRecallDataManager::get_TDR(idx_t q_id) {

        float found_sum = 0.0f;
        float gt_sum = 0.0f;
        int valid = 0;

        for (idx_t i = 0; i < k; i++) {

            float f = distances[q_id * k + i];
            float g = gt_dist[q_id * k + i];

            // 忽略 self-match
            if (f == 0.0f && g == 0.0f) continue;

            if (!std::isfinite(f) || !std::isfinite(g)) continue;

            found_sum += f;
            gt_sum += g;
            valid++;
        }

        if (valid == 0 || found_sum == 0.0f)
            return 1.0f;   // 或 0，看你定义；1=“完全匹配”的含义也说得通

        return gt_sum / found_sum;
    }

//    float DeclarativeRecallDataManager::get_TDR(idx_t q_id) {
//        float found_distances_sum = 0;
//        float gt_distances_sum = 0;
//
//        for (idx_t i = 0; i < k; i++) {
//            found_distances_sum += distances[q_id * k + i];
//            gt_distances_sum += gt_dist[q_id * k + i];
//        }
//
//        return gt_distances_sum / found_distances_sum;
//    }

    float DeclarativeRecallDataManager::get_RDE(idx_t q_id) {

        std::vector<float> found_distances(static_cast<size_t>(k));
        for (idx_t i = 0; i < k; i++) {
            found_distances[i] = distances[q_id * k + i];
        }

        std::sort(found_distances.begin(), found_distances.end());

        float RDE = 0.0f;
        int valid = 0;

        for (idx_t i = 0; i < k; i++) {
            float f = found_distances[i];
            float g = gt_dist[q_id * k + i];

            // 分母无效：为0 或 非有限数 → 跳过
            if (!std::isfinite(g) || g == 0.0f) {
                continue;
            }

            if (!std::isfinite(f)) {
                continue;
            }

            RDE += (f / g) - 1.0f;
            valid++;
        }

        if (valid > 0) {
            RDE /= valid;   // 只按有效项平均
        } else {
            RDE = 0.0f;     // fallback
        }

        return RDE;
    }

//    float DeclarativeRecallDataManager::get_RDE(idx_t q_id) {
//        std::vector<float> found_distances(static_cast<size_t>(k));
//        for (idx_t i = 0; i < k; i++) {
//            found_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
//        }
//
//        // sort by increasing distance to the query
//        std::sort(found_distances.begin(), found_distances.end());
//
//        float RDE = 0;
//
//        for (idx_t i = 0; i < k; i++) {
//            RDE += (found_distances[static_cast<size_t>(i)] / gt_dist[q_id * k + i]) - 1;
//        }
//
//        RDE /= k;
//
//        return RDE;
//    }

    float DeclarativeRecallDataManager::get_NRS(idx_t q_id) {
        std::vector<std::pair<float, idx_t>> distance_label_pairs;

        // Collect approximate distances and labels
        for (int i = 0; i < k; ++i) {
            distance_label_pairs.emplace_back(
                    distances[q_id * k + i], labels[q_id * k + i]);
        }

        // Sort the pairs based on distances
        std::sort(
                distance_label_pairs.begin(),
                distance_label_pairs.end(),
                [](const std::pair<float, idx_t> &a,
                   const std::pair<float, idx_t> &b) {
                    return a.first < b.first;
                });

        // Compute the rank sum for the ground-truth neighbors
        float rank_sum = 0;
        for (int i = 0; i < k; i++) {
            idx_t current_nn_id = distance_label_pairs[static_cast<size_t>(i)].second;

            // Find the rank of the current nearest neighbor in the ground truth
            idx_t gt_position = -1;
            for (int j = 0; j < k_all; j++) {
                if (gt_for_all_k[q_id * k_all + j] == current_nn_id) {
                    gt_position = j + 1;  // rank is 1-based
                    break;
                }
            }

            // If a ground-truth neighbor is missing, return -1 (undefined)
            if (gt_position == -1) {
                //更改惩罚为 k_all + 1;
                gt_position = k_all + 1;
//                return -1.0;
            }

            rank_sum += gt_position;
        }

        // Calculate max and min possible rank sums
        float max_rank_sum = k * (k + 1) / 2.0;

        // Normalize the rank sum
        float NRS = 1 - ((rank_sum - k) / (max_rank_sum - k));

        return NRS;
    }

// DeclarativeRecallDataCollectorHNSW
    DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW() {}

    DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW(
            DeclarativeRecallDataManager data_manager,
            int logging_interval,
            int dist_early_stop_threshold)
            : data_manager(data_manager),
              logging_interval(logging_interval),
              dist_early_stop_threshold(dist_early_stop_threshold) {
        if (dist_early_stop_threshold > 0) {
            do_naive_early_stop = true;
        }
    }

// DeclarativeRecallDataCollectorIVF
    void DeclarativeRecallDataCollectorIVF::init_log_file() {
        if (!dataManager.log_filename) {
            return;
        }

        log_file = fopen(dataManager.log_filename, "w");
        if (!log_file) {
            // printf("Error opening recall data log file %s\n",
            // dataManager.log_filename);
            perror("Error opening recall data log file");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "percentile_25,");
        fprintf(log_file, "percentile_50,");
        fprintf(log_file, "percentile_75,");
        fprintf(log_file, "percentile_95,");

        fprintf(log_file, "variance,");

        // New
        fprintf(log_file, "std,");
        fprintf(log_file, "range,");
        fprintf(log_file, "skewness,");
        fprintf(log_file, "kurtosis,");
        fprintf(log_file, "energy,");
        // New end

        fprintf(log_file, "nn10_dist,");
        fprintf(log_file, "nn_to_first,");
        fprintf(log_file, "nn10_to_first,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        /*if (include_data_dimensions){
            for (int i = 0; i < dataManager.d; i++) {
                fprintf(log_file, "d%d,", i);
            }
        }*/
        fprintf(log_file, "feats_collect_time_ms,");

        // Target
        fprintf(log_file, "r\n");
    }

    void DeclarativeRecallDataCollectorIVF::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

    void DeclarativeRecallDataCollectorIVF::append_to_log(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            float recall_k) {
        if (!log_file || total_insertions < dataManager.k) {
            return;
        }

        // Standard values
        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);
        fprintf(log_file, "%f,", first_nn_dis);

        double feature_collection_time_start = dataManager.elapsed_secs();

        // version 1 for distances
        float nn_dist = dataManager.get_nearest_dist_of_query(query_idx);
        float avg_dist = dataManager.get_avg_dist_of_query(query_idx);
        float furthest_dist = dataManager.get_furthest_dist_of_query(query_idx);

        fprintf(log_file, "%f,", nn_dist);
        fprintf(log_file, "%f,", avg_dist);
        fprintf(log_file, "%f,", furthest_dist);

        float percentile_25 =
                dataManager.get_percentile_of_query(query_idx, 0.25);
        float percentile_50 =
                dataManager.get_percentile_of_query(query_idx, 0.50);
        float percentile_75 =
                dataManager.get_percentile_of_query(query_idx, 0.75);
        float percentile_95 =
                dataManager.get_percentile_of_query(query_idx, 0.95);

        fprintf(log_file, "%f,", percentile_25);
        fprintf(log_file, "%f,", percentile_50);
        fprintf(log_file, "%f,", percentile_75);
        fprintf(log_file, "%f,", percentile_95);

        float variance = dataManager.get_variance_of_query(query_idx);
        fprintf(log_file, "%f,", variance);

        // New includes start
        float stdv = std::sqrt(variance);
        float range = furthest_dist - nn_dist;
        float skewness = dataManager.get_skewness_of_query(query_idx);
        float kurtosis = dataManager.get_kurtosis_of_query(query_idx);
        float energy = dataManager.get_energy_of_query(query_idx);
        fprintf(log_file, "%f,", stdv);
        fprintf(log_file, "%f,", range);
        fprintf(log_file, "%f,", skewness);
        fprintf(log_file, "%f,", kurtosis);
        fprintf(log_file, "%f,", energy);
        // New includes end

        // CMU features
        float dist_10 = -1;
        float dnn_to_dstart = -1;
        float d10_to_dstart = -1;

        if (dataManager.k >= 10) {
            dist_10 = dataManager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        if (first_nn_dis > 0) {
            dnn_to_dstart = nn_dist / first_nn_dis;
        }

        if (dist_10 != -1 && first_nn_dis > 0) {
            d10_to_dstart = dist_10 / first_nn_dis;
        }

        fprintf(log_file, "%f,", dist_10);
        fprintf(log_file, "%f,", dnn_to_dstart);
        fprintf(log_file, "%f,", d10_to_dstart);
        //

        // version 2 for distances
        // int all_result_set_feats = 11;
        // float dist_feats[all_result_set_feats];
        // get_rset_feats_of_query(query_idx, dist_feats, first_nn_dis);

        // for (int i = 0; i < all_result_set_feats; i++) {
        //     fprintf(log_file, "%f,", dist_feats[i]);
        // }

        // distance from query to medoid
        // float distance_from_medoid =
        //        dataManager.get_dist_of_query_to_medoid(query_idx);
        // fprintf(log_file, "%f,", distance_from_medoid);
        //

        float RDE = dataManager.get_RDE(query_idx);
        float TDR = dataManager.get_TDR(query_idx);
        float NRS = dataManager.get_NRS(query_idx);
        fprintf(log_file, "%f,", RDE);
        fprintf(log_file, "%f,", TDR);
        fprintf(log_file, "%f,", NRS);

        // if (include_data_dimensions){
        //    for (int i = 0; i < dataManager.d; i++) {
        //       fprintf(log_file, "%f,", dataManager.queries[query_idx * dataManager.d + i]);
        //    }
        //}

        // float query_dim_stats[dataManager.summary_stats_num];
        // dataManager.get_precomputed_query_stats(query_idx, query_dim_stats);
        // for (int i = 0; i < dataManager.summary_stats_num; i++) {
        //     fprintf(log_file, "%f,", query_dim_stats[i]);
        // }

        double feature_collection_time_end = dataManager.elapsed_secs();
        double feature_collection_time =
                (feature_collection_time_end - feature_collection_time_start) *
                1000;
        fprintf(log_file, "%f,", feature_collection_time);

        // Target
        fprintf(log_file, "%f\n", recall_k);
    }

    void DeclarativeRecallDataCollectorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            // printf("Error opening recall data log file %s\n",
            // dataManager.log_filename);
            perror("Error opening recall data log file");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");
        fprintf(log_file, "visited_points,");
        fprintf(log_file, "duration,");
        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "percentile_25,");
        fprintf(log_file, "percentile_50,");
        fprintf(log_file, "percentile_75,");
        fprintf(log_file, "percentile_95,");

        fprintf(log_file, "variance,");

        // New
        fprintf(log_file, "std,");
        fprintf(log_file, "range,");
        fprintf(log_file, "skewness,");
        fprintf(log_file, "kurtosis,");
        fprintf(log_file, "energy,");
        // New end

        fprintf(log_file, "nn10_dist,");
        fprintf(log_file, "nn_to_first,");
        fprintf(log_file, "nn10_to_first,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "feats_collect_time_ms,");

        // Target
        fprintf(log_file, "r\n");
    }

    void DeclarativeRecallDataCollectorHNSW::init_CNS_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            // printf("Error opening recall data log file %s\n",
            // dataManager.log_filename);
            perror("Error opening recall data log file");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");
        fprintf(log_file, "first_nn_dist,");
        fprintf(log_file, "visited_points,");
        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

//        fprintf(log_file, "percentile_25,");
//        fprintf(log_file, "percentile_50,");
//        fprintf(log_file, "percentile_75,");
//        fprintf(log_file, "percentile_95,");

        fprintf(log_file, "variance,");

        // New
        fprintf(log_file, "std,");
        fprintf(log_file, "range,");
        fprintf(log_file, "skewness,");
        fprintf(log_file, "kurtosis,");
        fprintf(log_file, "energy,");
        // New end

        fprintf(log_file, "nn10_dist,");
        fprintf(log_file, "nn_to_first,");
        fprintf(log_file, "nn10_to_first,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "degree_0,");
        fprintf(log_file, "avg_degree,");
        fprintf(log_file, "avg_hop,");

        //密度特征（平均距离的倒数）
        fprintf(log_file, "density,");
        //入口点到查询点距离 与入口点距离均值之间的比值特征
        fprintf(log_file, "entry_query_dist_ratio,");
        //距离分布熵
        fprintf(log_file, "dist_distribution_entropy,");
        //余弦特征，均值，方差、方向熵
        fprintf(log_file, "cosine_mean,");
        fprintf(log_file, "cosine_variance,");
        fprintf(log_file, "cosine_direction_entropy,");

        fprintf(log_file, "feats_collect_time_ms,");
        // Target
        fprintf(log_file, "r\n");

    }

    void DeclarativeRecallDataCollectorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

    void DeclarativeRecallDataCollectorHNSW::append_to_log(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            float recall_k,
            int visited_points) {
        if (!log_file || total_insertions < data_manager.k) {
            return;
        }

        // Standard values
        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);
        fprintf(log_file, "%f,", first_nn_dis);
        fprintf(log_file, "%d,", visited_points);
        int duration=0;
        fprintf(log_file, "%d,", duration);

        double feature_collection_time_start = data_manager.elapsed_secs();

        // version 1 for distances
        float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);

        fprintf(log_file, "%f,", nn_dist);
        fprintf(log_file, "%f,", avg_dist);
        fprintf(log_file, "%f,", furthest_dist);

        float percentile_25 =
                data_manager.get_percentile_of_query(query_idx, 0.25);
        float percentile_50 =
                data_manager.get_percentile_of_query(query_idx, 0.50);
        float percentile_75 =
                data_manager.get_percentile_of_query(query_idx, 0.75);
        float percentile_95 =
                data_manager.get_percentile_of_query(query_idx, 0.95);

        fprintf(log_file, "%f,", percentile_25);
        fprintf(log_file, "%f,", percentile_50);
        fprintf(log_file, "%f,", percentile_75);
        fprintf(log_file, "%f,", percentile_95);

        float variance = data_manager.get_variance_of_query(query_idx);
        fprintf(log_file, "%f,", variance);

        /* New includes start */
        float stdv = std::sqrt(variance);
        float range = furthest_dist - nn_dist;
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);
        fprintf(log_file, "%f,", stdv);
        fprintf(log_file, "%f,", range);
        fprintf(log_file, "%f,", skewness);
        fprintf(log_file, "%f,", kurtosis);
        fprintf(log_file, "%f,", energy);
        /* New includes end */

        // CMU features
        float dist_10 = -1;
        float dnn_to_dstart = -1;
        float d10_to_dstart = -1;

        if (data_manager.k >= 10) {
            dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        if (first_nn_dis > 0) {
            dnn_to_dstart = nn_dist / first_nn_dis;
        }

        if (dist_10 != -1 && first_nn_dis > 0) {
            d10_to_dstart = dist_10 / first_nn_dis;
        }

        fprintf(log_file, "%f,", dist_10);
        fprintf(log_file, "%f,", dnn_to_dstart);
        fprintf(log_file, "%f,", d10_to_dstart);
        //

        // version 2 for distances
        // int all_result_set_feats = 11;
        // float dist_feats[all_result_set_feats];
        // get_rset_feats_of_query(query_idx, dist_feats, first_nn_dis);

        // for (int i = 0; i < all_result_set_feats; i++) {
        //     fprintf(log_file, "%f,", dist_feats[i]);
        // }

        // distance from query to medoid
        // float distance_from_medoid =
        //        dataManager.get_dist_of_query_to_medoid(query_idx);
        // fprintf(log_file, "%f,", distance_from_medoid);
        //

        float RDE = data_manager.get_RDE(query_idx);
        float TDR = data_manager.get_TDR(query_idx);
        float NRS = data_manager.get_NRS(query_idx);
        fprintf(log_file, "%f,", RDE);
        fprintf(log_file, "%f,", TDR);
        fprintf(log_file, "%f,", NRS);

        // if (include_data_dimensions){
        //    for (int i = 0; i < dataManager.d; i++) {
        //       fprintf(log_file, "%f,", dataManager.queries[query_idx * dataManager.d + i]);
        //    }
        //}

        double feature_collection_time_end = data_manager.elapsed_secs();
        double feature_collection_time =
                (feature_collection_time_end - feature_collection_time_start) *
                1000;
        fprintf(log_file, "%f,", feature_collection_time);

        // Target
        fprintf(log_file, "%f\n", recall_k);
    }

    std::string DeclarativeRecallDataCollectorHNSW::get_observation_data_str(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            float recall_k,
            int visited_points,
            int duration) {
        std::string observation_data = "";

        if (!log_file || total_insertions < data_manager.k) {
            return observation_data;
        }

        double feature_collection_time_start = data_manager.elapsed_secs();


        observation_data += std::to_string(query_idx) + ",";
        observation_data += std::to_string(nstep) + ",";
        observation_data += std::to_string(ndis) + ",";
        observation_data += std::to_string(elapsed * 1000) + ",";
        observation_data += std::to_string(total_insertions) + ",";

        float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float visited_points = furthest_dist / d0;

        observation_data += std::to_string(first_nn_dis) + ",";
        observation_data += std::to_string(visited_points) + ",";
        observation_data += std::to_string(duration) + ",";

        observation_data += std::to_string(nn_dist) + ",";
        observation_data += std::to_string(avg_dist) + ",";
        observation_data += std::to_string(furthest_dist) + ",";

        float perc_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
        float perc_50 = data_manager.get_percentile_of_query(query_idx, 0.50);
        float perc_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
        float perc_95 = data_manager.get_percentile_of_query(query_idx, 0.95);
        observation_data += std::to_string(perc_25) + ",";
        observation_data += std::to_string(perc_50) + ",";
        observation_data += std::to_string(perc_75) + ",";
        observation_data += std::to_string(perc_95) + ",";

        float variance = data_manager.get_variance_of_query(query_idx);
        observation_data += std::to_string(variance) + ",";

        /* New includes start */
        float stdv = std::sqrt(variance);
        float range = furthest_dist - nn_dist;
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);
        observation_data += std::to_string(stdv) + ",";
        observation_data += std::to_string(range) + ",";
        observation_data += std::to_string(skewness) + ",";
        observation_data += std::to_string(kurtosis) + ",";
        observation_data += std::to_string(energy) + ",";
        /* New includes end */

        // LAET features
        float dist_10 = -1;
        float dnn_to_dstart = -1;
        float d10_to_dstart = -1;

        if (data_manager.k >= 10) {
            dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        if (first_nn_dis > 0) {
            dnn_to_dstart = nn_dist / first_nn_dis;
        }

        if (dist_10 != -1 && first_nn_dis > 0) {
            d10_to_dstart = dist_10 / first_nn_dis;
        }

        observation_data += std::to_string(dist_10) + ",";
        observation_data += std::to_string(dnn_to_dstart) + ",";
        observation_data += std::to_string(d10_to_dstart) + ",";

        // add the quality approximation measures
        float RDE = data_manager.get_RDE(query_idx);
        float TDR = data_manager.get_TDR(query_idx);
        float NRS = data_manager.get_NRS(query_idx);
        observation_data += std::to_string(RDE) + ",";
        observation_data += std::to_string(TDR) + ",";
        observation_data += std::to_string(NRS) + ",";

        double feature_collection_time_end = data_manager.elapsed_secs();
        double feature_collection_time =
                (feature_collection_time_end - feature_collection_time_start) *
                1000;
        observation_data += std::to_string(feature_collection_time) + ",";

        observation_data += std::to_string(recall_k) + "\n";

        return observation_data;
    }

    std::string DeclarativeRecallDataCollectorHNSW::get_observation_CNS_data_str(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            float recall_k,
            int visited_points,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        std::string observation_data = "";

        if (!log_file || total_insertions < data_manager.k) {
            return observation_data;
        }

        double feature_collection_time_start = data_manager.elapsed_secs();


        observation_data += std::to_string(query_idx) + ",";
        observation_data += std::to_string(nstep) + ",";
        observation_data += std::to_string(ndis) + ",";
        observation_data += std::to_string(elapsed * 1000) + ",";
        observation_data += std::to_string(total_insertions) + ",";
        observation_data += std::to_string(first_nn_dis) + ",";

        float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float visited_points = furthest_dist / d0;
//        float visited_points_k = static_cast<float>(visited_points) / data_manager.k;
        observation_data += std::to_string(visited_points) + ",";
        observation_data += std::to_string(nn_dist) + ",";
        observation_data += std::to_string(avg_dist) + ",";
        observation_data += std::to_string(furthest_dist) + ",";

//        float perc_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float perc_50 = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float perc_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float perc_95 = data_manager.get_percentile_of_query(query_idx, 0.95);
//        observation_data += std::to_string(perc_25) + ",";
//        observation_data += std::to_string(perc_50) + ",";
//        observation_data += std::to_string(perc_75) + ",";
//        observation_data += std::to_string(perc_95) + ",";

        float variance = data_manager.get_variance_of_query(query_idx);
        observation_data += std::to_string(variance) + ",";

        /* New includes start */
        float stdv = std::sqrt(variance);
        float range = furthest_dist - nn_dist;
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);
        observation_data += std::to_string(stdv) + ",";
        observation_data += std::to_string(range) + ",";
        observation_data += std::to_string(skewness) + ",";
        observation_data += std::to_string(kurtosis) + ",";
        observation_data += std::to_string(energy) + ",";
        /* New includes end */

        // LAET features
        float dist_10 = -1;
        float dnn_to_dstart = -1;
        float d10_to_dstart = -1;

        if (data_manager.k >= 10) {
            dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        if (first_nn_dis > 0) {
            dnn_to_dstart = nn_dist / first_nn_dis;
        }

        if (dist_10 != -1 && first_nn_dis > 0) {
            d10_to_dstart = dist_10 / first_nn_dis;
        }

        observation_data += std::to_string(dist_10) + ",";
        observation_data += std::to_string(dnn_to_dstart) + ",";
        observation_data += std::to_string(d10_to_dstart) + ",";

        // add the quality approximation measures
        float RDE = data_manager.get_RDE(query_idx);
        float TDR = data_manager.get_TDR(query_idx);
        float NRS = data_manager.get_NRS(query_idx);
        observation_data += std::to_string(RDE) + ",";
        observation_data += std::to_string(TDR) + ",";
        observation_data += std::to_string(NRS) + ",";

        double feature_collection_time_end = data_manager.elapsed_secs();
        double feature_collection_time =
                (feature_collection_time_end - feature_collection_time_start) *
                1000;

        //0号位置入度特征
        idx_t nn_id = data_manager.get_nearest_id_of_query(query_idx);
        int degree_0 = data_manager.get_out_degree(nn_id);
        observation_data += std::to_string(degree_0) + ",";
        //平均入度特征
        float avg_degree = data_manager.get_avg_degree_of_query(query_idx);
        observation_data += std::to_string(avg_degree) + ",";
        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        observation_data += std::to_string(avg_hop) + ",";

        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        observation_data += std::to_string(density) + ",";
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        observation_data += std::to_string(entry_query_dist_ratio) + ",";
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        observation_data += std::to_string(dist_distribution_entropy) + ",";
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);
        //余弦均值特征
        observation_data += std::to_string(cs.mean) + ",";
        //余弦方差特征
        observation_data += std::to_string(cs.variance) + ",";
        //余弦方向熵特征
        observation_data += std::to_string(cs.direction_entropy) + ",";

        observation_data += std::to_string(feature_collection_time) + ",";

        observation_data += std::to_string(recall_k) + "\n";

        return observation_data;
    }

    void DeclarativeRecallDataCollectorHNSW::flush_observation_to_log(std::string observation_data) {
        if (!log_file) {
            return;
        }

        fprintf(log_file, "%s", observation_data.c_str());
    }

    void DeclarativeRecallDataCollectorHNSW::flush_all_observations_to_log(std::string *observations, int n) {
        if (!log_file) {
            return;
        }
        for (int i = 0; i < n; i++) {
            fprintf(log_file, "%s", observations[i].c_str());
        }
    }

// DARTHPredictorHNSW
    DARTHPredictorHNSW::DARTHPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            double target_recall,
            int initial_prediction_interval,
            int min_prediction_interval,
            bool per_prediction_logging,
            char *predictor_model_path)
            : data_manager(data_manager),
              target_recall(target_recall),
              initial_prediction_interval(initial_prediction_interval),
              min_prediction_interval(min_prediction_interval),
              per_prediction_logging(per_prediction_logging) {//构造函数方法 将参数复制到成员变量
        int out_iterations;
        int result = LGBM_BoosterCreateFromModelfile(
                predictor_model_path, &out_iterations, &booster);
        if (result != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }

//创建日志文件并写表头
    void DARTHPredictorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            printf("Error opening recall data log file\n");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "variance,");
        fprintf(log_file, "median,");

        fprintf(log_file, "percentile_25,");
        fprintf(log_file, "percentile_75,");

        fprintf(log_file, "r_current_interval,");
        fprintf(log_file, "r_predictor_calls,");
        fprintf(log_file, "r_predictor_time_ms,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r_actual,");
        fprintf(log_file, "r_predicted\n");
    }

    void DARTHPredictorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

//核心方法，这个函数基于当前 HNSW 搜索状态预测 recall，用于判断是否继续搜索。
    float DARTHPredictorHNSW::predict_recall(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
            int *prediction_interval,
            double *predictor_time) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        // query_predictor_calls++;

        double predictor_start_time = data_manager.elapsed_secs();

        float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);

        const int num_feats = 11;
        const double data[11] = {
                (double) nstep,
                (double) ndis,
                (double) total_insertions,
                (double) first_nn_dis,
                nn_dist,
                avg_dist,
                furthest_dist,
                variance,
                median,
                percentile_25,
                percentile_75};

        double out_result[1];
        long int out_len;

        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                data,
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_recall = (float) out_result[0];
        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

        if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
            predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
        }

        float recall_diff = target_recall - predicted_recall;
        *prediction_interval =
                min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);

            fprintf(log_file, "%f,", first_nn_dis);
            fprintf(log_file, "%f,", nn_dist);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);

            fprintf(log_file, "%f,", variance);
            fprintf(log_file, "%f,", median);

            fprintf(log_file, "%f,", percentile_25);
            fprintf(log_file, "%f,", percentile_75);

            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_recall);
        }

        return predicted_recall;
    }

    void DARTHPredictorHNSW::log_final_recall_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            double last_predicted_recall,
            int prediction_interval,
            int query_predictor_calls,
            double predictor_time) {
        if (!log_file || per_prediction_logging) {
            return;
        }

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);

        fprintf(log_file, "%f,", first_nn_dis);

        fprintf(log_file, "%f,", data_manager.get_nearest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_avg_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_furthest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_variance_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.50));
        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.25));
        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.75));

        fprintf(log_file, "%d,", prediction_interval);
        fprintf(log_file, "%d,", query_predictor_calls);
        fprintf(log_file, "%f,", predictor_time * 1000);

        fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
        fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
        fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

        fprintf(log_file, "%f,", data_manager.get_recallk(query_idx));
        fprintf(log_file, "%f\n", last_predicted_recall);
    }

    RAETPredictorHNSW::RAETPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            double target_recall,
            int stability_times,
            int stability_times_r1,
            bool per_prediction_logging,
            char *predictor_model_path)
            : data_manager(data_manager),
              target_recall(target_recall),
              stability_times(stability_times),
              stability_times_r1(stability_times_r1),
              per_prediction_logging(per_prediction_logging) {//构造函数方法 将参数复制到成员变量
        int out_iterations;
        int result = LGBM_BoosterCreateFromModelfile(
                predictor_model_path, &out_iterations, &booster);
        if (result != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }

    //创建日志文件并写表头
    void RAETPredictorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            printf("Error opening recall data log file\n");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");
        fprintf(log_file, "visited_points,");
        fprintf(log_file, "duration,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "variance,");
//        fprintf(log_file, "median,");
//
//        fprintf(log_file, "percentile_25,");
//        fprintf(log_file, "percentile_75,");

//        fprintf(log_file, "r_current_interval,");
        fprintf(log_file, "r_predictor_calls,");
        fprintf(log_file, "r_predictor_time_ms,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r_actual,");
        fprintf(log_file, "r_predicted\n");
    }

    void RAETPredictorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

//核心方法，这个函数基于当前 HNSW 搜索状态预测 recall，用于判断是否继续搜索。
    float RAETPredictorHNSW::predict_recall(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
//            int *prediction_interval,
            double *predictor_time,
            int visited_points,
            int duration) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);
//        //原本9个特征
//        int num_nonquery_feats = 9;
//        int query_feats = data_manager.d;
//        int total_feats = num_nonquery_feats + query_feats;
//        std::vector<double> feats(static_cast<size_t>(total_feats));
//
//        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
//        float variance = data_manager.get_variance_of_query(query_idx);
////        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
////        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
////        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
////        float visited_points = furthest_dist / d0;
//
//        feats[0] = nstep;//跳数
//        feats[1] = ndis;//
//        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
//
//        feats[3] = d_start;//底层入口点
//        feats[4] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
//
//        feats[5] = d_1st;//0号位置数据点
//        feats[6] = furthest_dist;//k-1号位置数据点
//
//        feats[7] = avg_dist;//距离均值
//        feats[8] = variance;//距离方差

//        //删除插入到前k位置数据点数量特征
//        int num_nonquery_feats = 8;
//        int query_feats = data_manager.d;
//        int total_feats = num_nonquery_feats + query_feats;
//        std::vector<double> feats(static_cast<size_t>(total_feats));
//
//        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
//        float variance = data_manager.get_variance_of_query(query_idx);
////        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
////        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
////        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
////        float visited_points = furthest_dist / d0;
//
//        feats[0] = nstep;//跳数
//        feats[1] = ndis;//
////        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
//
//        feats[2] = d_start;//底层入口点
//        feats[3] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
//
//        feats[4] = d_1st;//0号位置数据点
//        feats[5] = furthest_dist;//k-1号位置数据点
//
//        feats[6] = avg_dist;//距离均值
//        feats[7] = variance;//距离方差

//        //删除入口点距离特征(正确方法开始)
//        int num_nonquery_feats = 8;
//        int query_feats = data_manager.d;
//        int total_feats = num_nonquery_feats + query_feats;
//        std::vector<double> feats(static_cast<size_t>(total_feats));
//
//        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
//        float variance = data_manager.get_variance_of_query(query_idx);
////        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
////        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
////        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
////        float visited_points = furthest_dist / d0;
//
//        feats[0] = nstep;//跳数
//        feats[1] = ndis;//
//        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
//
////        feats[3] = d_start;//底层入口点
//        feats[3] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
//
//        feats[4] = d_1st;//0号位置数据点
//        feats[5] = furthest_dist;//k-1号位置数据点
//
//        feats[6] = avg_dist;//距离均值
//        feats[7] = variance;//距离方差
//
////        //删除入口点距离特征和插入到前k位置数量特征
////        int num_nonquery_feats = 7;
////        int query_feats = data_manager.d;
////        int total_feats = num_nonquery_feats + query_feats;
////        std::vector<double> feats(static_cast<size_t>(total_feats));
////
////        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
////        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
////        float variance = data_manager.get_variance_of_query(query_idx);
//////        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
//////        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//////        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//////        float visited_points = furthest_dist / d0;
////
////        feats[0] = nstep;//跳数
////        feats[1] = ndis;//
//////        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
////
//////        feats[3] = d_start;//底层入口点
////
////        feats[2] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
////        feats[3] = d_1st;//0号位置数据点
////
////        feats[4] = furthest_dist;//k-1号位置数据点
////
////        feats[5] = avg_dist;//距离均值
////        feats[6] = variance;//距离方差
//
//
//        for (int i = 0; i < data_manager.d; i++) {
//            feats[static_cast<size_t>(num_nonquery_feats + i)] =
//                    data_manager.queries[query_idx * data_manager.d + i];
//        }
//
//        int num_rows = 1;
//        int num_feats = total_feats;
        //删除入口点距离特征(正确方法结束)

        //删除入口点距离特征
        int total_feats = 9;
//        int query_feats = data_manager.d;
//        int total_feats = num_nonquery_feats + query_feats;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
//        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float visited_points = furthest_dist / d0;
        float visited_points_k = static_cast<float>(visited_points) / data_manager.k;

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

//        feats[3] = d_start;//底层入口点
        feats[3] = visited_points_k;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
        feats[4] = duration;
        feats[5] = d_1st;//0号位置数据点
        feats[6] = furthest_dist;//k-1号位置数据点

        feats[7] = avg_dist;//距离均值
        feats[8] = variance;//距离方差

        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

//        //删除查询点向量特征
//        int num_nonquery_feats = 9;
//        int query_feats = data_manager.d;
//        std::vector<double> feats(static_cast<size_t>(num_nonquery_feats));
//
//        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
//        float variance = data_manager.get_variance_of_query(query_idx);
////        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
////        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
////        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//
//        feats[0] = nstep;//跳数
//        feats[1] = ndis;//
//        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
//
//        feats[3] = d_start;//底层入口点
//        feats[4] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
//
//        feats[5] = d_1st;//0号位置数据点
//        feats[6] = furthest_dist;//k-1号位置数据点
//
//        feats[7] = avg_dist;//距离均值
//        feats[8] = variance;//距离方差
//
//        int num_rows = 1;
//        int num_feats = num_nonquery_feats;
//
//        double out_result[1];
//        long int out_len;
//        int res = LGBM_BoosterPredictForMatSingleRow(
//                booster,
//                feats.data(),
//                C_API_DTYPE_FLOAT64,
//                num_feats,
//                1,
//                C_API_PREDICT_NORMAL,
//                0,
//                -1,
//                "",
//                &out_len,
//                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_recall = (float) out_result[0];
        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

        if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
            predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
        }

//        float recall_diff = target_recall - predicted_recall;
//        *prediction_interval =min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%f,", first_nn_dis);

            fprintf(log_file, "%d,", visited_points);
            fprintf(log_file, "%d,", duration);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_recall);
        }

        return predicted_recall;
    }

    void RAETPredictorHNSW::log_final_recall_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            double last_predicted_recall,
//            int prediction_interval,
            int query_predictor_calls,
            double predictor_time,
            int visited_points,
            int duration) {
        if (!log_file || per_prediction_logging) {
            return;
        }

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);


        fprintf(log_file, "%f,", first_nn_dis);
        fprintf(log_file, "%d,", visited_points);
        fprintf(log_file, "%d,", duration);

        fprintf(log_file, "%f,", data_manager.get_nearest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_avg_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_furthest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_variance_of_query(query_idx));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.50));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.25));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.75));

//        fprintf(log_file, "%d,", prediction_interval);
        fprintf(log_file, "%d,", query_predictor_calls);
        fprintf(log_file, "%f,", predictor_time * 1000);

        fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
        fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
        fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

        fprintf(log_file, "%f,", data_manager.get_recallk(query_idx));
        fprintf(log_file, "%f\n", last_predicted_recall);
    }

    RAETCNSPredictorHNSW::RAETCNSPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            double target_recall,
//            int stability_times,
            bool per_prediction_logging,
            char *predictor_model_path)
            : data_manager(data_manager),
              target_recall(target_recall),
              per_prediction_logging(per_prediction_logging) {//构造函数方法 将参数复制到成员变量
        int out_iterations;
        int result = LGBM_BoosterCreateFromModelfile(
                predictor_model_path, &out_iterations, &booster);
        if (result != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }


    //创建日志文件并写表头
    void RAETCNSPredictorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            printf("Error opening recall data log file\n");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");
//        fprintf(log_file, "visited_points,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");
        fprintf(log_file, "variance,");

        fprintf(log_file, "avg_hop,");
        fprintf(log_file, "density,");
        fprintf(log_file, "entry_query_dist_ratio,");
        fprintf(log_file, "dist_distribution_entropy,");
        fprintf(log_file, "skewness,");
        fprintf(log_file, "kurtosis,");
        fprintf(log_file, "energy,");
        fprintf(log_file, "cosine_mean,");
        fprintf(log_file, "cosine_variance,");
        fprintf(log_file, "cosine_direction_entropy,");
//        fprintf(log_file, "median,");
//
//        fprintf(log_file, "percentile_25,");
//        fprintf(log_file, "percentile_75,");

//        fprintf(log_file, "r_current_interval,");
        fprintf(log_file, "r_predictor_calls,");
        fprintf(log_file, "r_predictor_time_ms,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r,");
        fprintf(log_file, "CNS_predicted\n");
    }

    void RAETCNSPredictorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

//核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETCNSPredictorHNSW::predict_CNS(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
//            int *prediction_interval,
            double *predictor_time,
            int visited_points) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 10;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats + query_feats;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
//        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float visited_points = furthest_dist / d0;
        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
        feats[3] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替

        feats[4] = d_start;//底层入口点
        feats[5] = d_1st;//0号位置数据点
        feats[6] = furthest_dist;//k-1号位置数据点

        feats[7] = avg_dist;//距离均值
        feats[8] = variance;//距离方差
        feats[9] = avg_hop;//平均跳数特征

        for (int i = 0; i < data_manager.d; i++) {
            feats[static_cast<size_t>(num_nonquery_feats + i)] =
                    data_manager.queries[query_idx * data_manager.d + i];
        }

        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_CNS = (float) out_result[0];
//        predicted_CNS = std::min(1.0f, std::max(0.0f, predicted_CNS));  // Make sure recall is in [0, 1]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_CNS);
        }

        return predicted_CNS;
    }

    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETCNSPredictorHNSW::predict_CNS_test(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 18;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats + query_feats;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
//        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float visited_points = furthest_dist / d0;


        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);



        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点
        feats[6] = avg_dist;//距离均值
        feats[7] = variance;//距离方差
        feats[8] = avg_hop;//平均跳数特征
        feats[9] = density;//密度特征特征
        feats[10] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[11] = dist_distribution_entropy;//距离分布熵
        feats[12] = skewness;
        feats[13] = kurtosis;
        feats[14] = energy;
        //密度特征
        feats[15] = cs.mean;
        feats[16] = cs.variance;
        feats[17] = cs.direction_entropy;

        for (int i = 0; i < data_manager.d; i++) {
            feats[static_cast<size_t>(num_nonquery_feats + i)] =
                    data_manager.queries[query_idx * data_manager.d + i];
        }

        int num_rows = 1;
        int num_feats = total_feats;
//        double predictor_start_time = data_manager.elapsed_secs();

//        double predictor_end_time = data_manager.elapsed_secs();

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

//        double raw_pred = out_result[0];
//        /* 1️⃣ safety floor before scaling */
//        raw_pred = std::max<double>(raw_pred, k);
//        /* 2️⃣ γ 校准（关键） */
//        int predicted_CNS = (int) std::ceil(1.0328 * raw_pred);
//        /* 3️⃣ safety floor & ceiling */
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        // LightGBM 输出的是 log1p(label)
//        double y_pred_log = out_result[0];
//        // 1️⃣ log → 原空间
//        double y_pred = std::expm1(y_pred_log);
//        // 2️⃣ 四舍五入 or 向上取整（efSearch 建议 ceil）
//        int predicted_CNS = static_cast<int>(std::ceil(y_pred));
//        // 3️⃣ 安全裁剪
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        int predicted_CNS = (int) out_result[0];

        //使用区域放大因子进行放大
//        if(predicted_CNS>150 && predicted_CNS<=160){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.017397));
//        }else if(predicted_CNS>160 && predicted_CNS<=170){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.051415));
//        }else if(predicted_CNS>170 && predicted_CNS<=180){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.069137));
//        }else if(predicted_CNS>180 && predicted_CNS<=190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.060578));
//        }else if(predicted_CNS>190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.030164));
//        }

        predicted_CNS = std::max(k, std::min(efSearch, predicted_CNS));  // Make sure recall is in [k, efSearch]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }

        return predicted_CNS;
    }
    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETCNSPredictorHNSW::predict_CNS_test_Noquery_Noske_Nodensty(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 14;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats ;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
//        float median = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float visited_points = furthest_dist / d0;


        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);



        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点
        feats[6] = avg_dist;//距离均值
        feats[7] = variance;//距离方差
        feats[8] = avg_hop;//平均跳数特征
//        feats[9] = density;//密度特征特征
        feats[9] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[10] = dist_distribution_entropy;//距离分布熵
//        feats[12] = skewness;
//        feats[13] = kurtosis;
//        feats[14] = energy;
        //密度特征
        feats[11] = cs.mean;
        feats[12] = cs.variance;
        feats[13] = cs.direction_entropy;


        int num_rows = 1;
        int num_feats = total_feats;
//        double predictor_start_time = data_manager.elapsed_secs();

//        double predictor_end_time = data_manager.elapsed_secs();

        double out_result[1];
        long int out_len;

        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);


        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

//        double raw_pred = out_result[0];
//        /* 1️⃣ safety floor before scaling */
//        raw_pred = std::max<double>(raw_pred, k);
//        /* 2️⃣ γ 校准（关键） */
//        int predicted_CNS = (int) std::ceil(1.0328 * raw_pred);
//        /* 3️⃣ safety floor & ceiling */
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        // LightGBM 输出的是 log1p(label)
//        double y_pred_log = out_result[0];
//        // 1️⃣ log → 原空间
//        double y_pred = std::expm1(y_pred_log);
//        // 2️⃣ 四舍五入 or 向上取整（efSearch 建议 ceil）
//        int predicted_CNS = static_cast<int>(std::ceil(y_pred));
//        // 3️⃣ 安全裁剪
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        int predicted_CNS = (int) out_result[0];

        //使用区域放大因子进行放大
//        if(predicted_CNS>150 && predicted_CNS<=160){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.017397));
//        }else if(predicted_CNS>160 && predicted_CNS<=170){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.051415));
//        }else if(predicted_CNS>170 && predicted_CNS<=180){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.069137));
//        }else if(predicted_CNS>180 && predicted_CNS<=190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.060578));
//        }else if(predicted_CNS>190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.030164));
//        }

        predicted_CNS = std::max(k, std::min(efSearch, predicted_CNS));  // Make sure recall is in [k, efSearch]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }

        return predicted_CNS;
    }

    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETCNSPredictorHNSW::predict_CNS_test_classification(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 18;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats + query_feats;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);

        feats[0]=nstep;
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量
//        feats[3] = visited_points;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点

        feats[6] = avg_dist;//距离均值
        feats[7]=variance;
        feats[8] = avg_hop;//平均跳数特征
        feats[9]=density;
        feats[10] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[11] = dist_distribution_entropy;//距离分布熵
        feats[12]=skewness;
        feats[13] = kurtosis;
        feats[14]=energy;
        //密度特征
        feats[15] = cs.mean;
        feats[16] = cs.variance;
        feats[17] = cs.direction_entropy;


        for (int i = 0; i < data_manager.d; i++) {
            feats[static_cast<size_t>(num_nonquery_feats + i)] =
                    data_manager.queries[query_idx * data_manager.d + i];
        }

        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        // probability of class=1
        double prob = out_result[0];

        // threshold = 0.5 (you can tune it)
        int cls = (prob >= 0.5) ? 1 : 0;

        // map to CNS
        int predicted_CNS = (cls == 0) ? k : efSearch;

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);

            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }

        return predicted_CNS;
    }

    void RAETCNSPredictorHNSW::log_final_recall_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
            double predictor_time,
            float best_dist_so_far,
            float entry_query_dist_ratio,
            int predicted_CNS) {
        if (!log_file || per_prediction_logging) {
            return;
        }
        double actual_recall_k = data_manager.get_recallk(query_idx);

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);
        fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

        fprintf(log_file, "%f,", d_1st);
        fprintf(log_file, "%f,", avg_dist);
        fprintf(log_file, "%f,", furthest_dist);
        fprintf(log_file, "%f,", variance);

        fprintf(log_file, "%f,", avg_hop);
        fprintf(log_file, "%f,", density);
        fprintf(log_file, "%f,", entry_query_dist_ratio);
        fprintf(log_file, "%f,", dist_distribution_entropy);
        fprintf(log_file, "%f,", skewness);
        fprintf(log_file, "%f,", kurtosis);
        fprintf(log_file, "%f,", energy);
        fprintf(log_file, "%f,", cs.mean);
        fprintf(log_file, "%f,", cs.variance);
        fprintf(log_file, "%f,", cs.direction_entropy);

        fprintf(log_file, "%d,", 1);
        fprintf(log_file, "%f,", predictor_time * 1000);

        fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
        fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
        fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

        fprintf(log_file, "%f,", actual_recall_k);
        fprintf(log_file, "%d\n", predicted_CNS);
    }
    RAETModelSelectPredictorHNSW::RAETModelSelectPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            double target_recall,
            int stability_times,
            int stability_times_r1,
            bool per_prediction_logging,
            char *predictor_model_path_recall,
            char *predictor_model_path_CNS,
            char *model_selector_path)
            : data_manager(data_manager),
              target_recall(target_recall),
              stability_times(stability_times),
              stability_times_r1(stability_times_r1),
              per_prediction_logging(per_prediction_logging) {//构造函数方法 将参数复制到成员变量
        int out_iterations;
        int result_recall = LGBM_BoosterCreateFromModelfile(
                predictor_model_path_recall, &out_iterations, &booster_recall);
//        double search_start_time = elapsed();
        int result_CNS = LGBM_BoosterCreateFromModelfile(
                predictor_model_path_CNS, &out_iterations, &booster_CNS);
        int result_model_selector = LGBM_BoosterCreateFromModelfile(
                model_selector_path, &out_iterations, &booster_selector);
//        double search_time = elapsed() - search_start_time;
//        printf("\n\n加载2个模型时间: %lfs,",search_time);
        if (result_recall != 0 && result_CNS != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }

    //创建日志文件并写表头
    void RAETModelSelectPredictorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            printf("Error opening recall data log file\n");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");
        fprintf(log_file, "visited_points,");
        fprintf(log_file, "duration,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "variance,");
//        fprintf(log_file, "median,");
//
//        fprintf(log_file, "percentile_25,");
//        fprintf(log_file, "percentile_75,");

//        fprintf(log_file, "r_current_interval,");
        fprintf(log_file, "r_predictor_calls,");
        fprintf(log_file, "r_predictor_time_ms,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r_actual,");
        fprintf(log_file, "r_predicted\n");
    }
    void RAETModelSelectPredictorHNSW::init_Select_Model_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            // printf("Error opening recall data log file %s\n",
            // dataManager.log_filename);
            perror("Error opening recall data log file");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");
        fprintf(log_file, "first_nn_dist,");
        fprintf(log_file, "visited_points,");
        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

//        fprintf(log_file, "percentile_25,");
//        fprintf(log_file, "percentile_50,");
//        fprintf(log_file, "percentile_75,");
//        fprintf(log_file, "percentile_95,");

        fprintf(log_file, "variance,");

        // New
        fprintf(log_file, "std,");
        fprintf(log_file, "range,");
        fprintf(log_file, "skewness,");
        fprintf(log_file, "kurtosis,");
        fprintf(log_file, "energy,");
        // New end

        fprintf(log_file, "nn10_dist,");
        fprintf(log_file, "nn_to_first,");
        fprintf(log_file, "nn10_to_first,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "degree_0,");
        fprintf(log_file, "avg_degree,");
        fprintf(log_file, "avg_hop,");

        //密度特征（平均距离的倒数）
        fprintf(log_file, "density,");
        //入口点到查询点距离 与入口点距离均值之间的比值特征
        fprintf(log_file, "entry_query_dist_ratio,");
        //距离分布熵
        fprintf(log_file, "dist_distribution_entropy,");
        //余弦特征，均值，方差、方向熵
        fprintf(log_file, "cosine_mean,");
        fprintf(log_file, "cosine_variance,");
        fprintf(log_file, "cosine_direction_entropy,");

        fprintf(log_file, "feats_collect_time_ms,");
        // Target
        fprintf(log_file, "r\n");

    }

    void RAETModelSelectPredictorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }
    std::string RAETModelSelectPredictorHNSW::get_observation_CNS_data_str(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            float recall_k,
            int visited_points,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        std::string observation_data = "";

        if (!log_file || total_insertions < data_manager.k) {
            return observation_data;
        }

        double feature_collection_time_start = data_manager.elapsed_secs();


        observation_data += std::to_string(query_idx) + ",";
        observation_data += std::to_string(nstep) + ",";
        observation_data += std::to_string(ndis) + ",";
        observation_data += std::to_string(elapsed * 1000) + ",";
        observation_data += std::to_string(total_insertions) + ",";
        observation_data += std::to_string(first_nn_dis) + ",";

        float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
//        float visited_points = furthest_dist / d0;
//        float visited_points_k = static_cast<float>(visited_points) / data_manager.k;
        observation_data += std::to_string(visited_points) + ",";
        observation_data += std::to_string(nn_dist) + ",";
        observation_data += std::to_string(avg_dist) + ",";
        observation_data += std::to_string(furthest_dist) + ",";

//        float perc_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
//        float perc_50 = data_manager.get_percentile_of_query(query_idx, 0.50);
//        float perc_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
//        float perc_95 = data_manager.get_percentile_of_query(query_idx, 0.95);
//        observation_data += std::to_string(perc_25) + ",";
//        observation_data += std::to_string(perc_50) + ",";
//        observation_data += std::to_string(perc_75) + ",";
//        observation_data += std::to_string(perc_95) + ",";

        float variance = data_manager.get_variance_of_query(query_idx);
        observation_data += std::to_string(variance) + ",";

        /* New includes start */
        float stdv = std::sqrt(variance);
        float range = furthest_dist - nn_dist;
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);
        observation_data += std::to_string(stdv) + ",";
        observation_data += std::to_string(range) + ",";
        observation_data += std::to_string(skewness) + ",";
        observation_data += std::to_string(kurtosis) + ",";
        observation_data += std::to_string(energy) + ",";
        /* New includes end */

        // LAET features
        float dist_10 = -1;
        float dnn_to_dstart = -1;
        float d10_to_dstart = -1;

        if (data_manager.k >= 10) {
            dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        if (first_nn_dis > 0) {
            dnn_to_dstart = nn_dist / first_nn_dis;
        }

        if (dist_10 != -1 && first_nn_dis > 0) {
            d10_to_dstart = dist_10 / first_nn_dis;
        }

        observation_data += std::to_string(dist_10) + ",";
        observation_data += std::to_string(dnn_to_dstart) + ",";
        observation_data += std::to_string(d10_to_dstart) + ",";

        // add the quality approximation measures
        float RDE = data_manager.get_RDE(query_idx);
        float TDR = data_manager.get_TDR(query_idx);
        float NRS = data_manager.get_NRS(query_idx);
        observation_data += std::to_string(RDE) + ",";
        observation_data += std::to_string(TDR) + ",";
        observation_data += std::to_string(NRS) + ",";

        double feature_collection_time_end = data_manager.elapsed_secs();
        double feature_collection_time =
                (feature_collection_time_end - feature_collection_time_start) *
                1000;

        //0号位置入度特征
        idx_t nn_id = data_manager.get_nearest_id_of_query(query_idx);
        int degree_0 = data_manager.get_out_degree(nn_id);
        observation_data += std::to_string(degree_0) + ",";
        //平均入度特征
        float avg_degree = data_manager.get_avg_degree_of_query(query_idx);
        observation_data += std::to_string(avg_degree) + ",";
        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        observation_data += std::to_string(avg_hop) + ",";

        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        observation_data += std::to_string(density) + ",";
        //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
        observation_data += std::to_string(entry_query_dist_ratio) + ",";
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        observation_data += std::to_string(dist_distribution_entropy) + ",";
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);
        //余弦均值特征
        observation_data += std::to_string(cs.mean) + ",";
        //余弦方差特征
        observation_data += std::to_string(cs.variance) + ",";
        //余弦方向熵特征
        observation_data += std::to_string(cs.direction_entropy) + ",";

        observation_data += std::to_string(feature_collection_time) + ",";

        observation_data += std::to_string(recall_k) + "\n";

        return observation_data;
    }

    void RAETModelSelectPredictorHNSW::flush_observation_to_log(std::string observation_data) {
        if (!log_file) {
            return;
        }

        fprintf(log_file, "%s", observation_data.c_str());
    }

    void RAETModelSelectPredictorHNSW::flush_all_observations_to_log(std::string *observations, int n) {
        if (!log_file) {
            return;
        }
        for (int i = 0; i < n; i++) {
            fprintf(log_file, "%s", observations[i].c_str());
        }
    }


//核心方法，这个函数基于当前 HNSW 搜索状态 判断是预测 recall还是预测CNS，用于判断是否继续搜索。
    float RAETModelSelectPredictorHNSW::predict_recall(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
//            int *prediction_interval,
            double *predictor_time,
            int visited_points,
            int duration,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
        float visited_points_k = static_cast<float>(visited_points) / data_manager.k;

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //删除入口点距离特征
        int total_feats = 9;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

//        feats[3] = d_start;//底层入口点
        feats[3] = visited_points_k;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
        feats[4] = duration;
        feats[5] = d_1st;//0号位置数据点
        feats[6] = furthest_dist;//k-1号位置数据点

        feats[7] = avg_dist;//距离均值
        feats[8] = variance;//距离方差

        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                //需要更改
                booster_recall,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_recall = (float) out_result[0];
        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

        if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
            predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
        }

//        float recall_diff = target_recall - predicted_recall;
//        *prediction_interval =min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%f,", first_nn_dis);

            fprintf(log_file, "%d,", visited_points);
            fprintf(log_file, "%d,", duration);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_recall);
        }

        return predicted_recall;
    }
    float RAETModelSelectPredictorHNSW::Model_Select_predict_recall(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
//            int *prediction_interval,
            double *predictor_time,
            int visited_points,
            int duration,
            float best_dist_so_far,
            float entry_query_dist_ratio) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);
        float visited_points_k = static_cast<float>(visited_points) / data_manager.k;

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //只第一次进行准确率预测时，进行一次模型选择器的预测
        if(query_predictor_calls==0){
            int selector_total_feats = 14;
            std::vector<double> selector_feats(static_cast<size_t>(selector_total_feats));

            selector_feats[0] = nstep;//跳数
            selector_feats[1] = ndis;//
            selector_feats[2] = total_insertions;//插入到CNS前k位置数据点数量

            selector_feats[3] = d_start;//底层入口点
            selector_feats[4] = d_1st;//0号位置数据点
            selector_feats[5] = furthest_dist;//k-1号位置数据点
            selector_feats[6] = avg_dist;//距离均值
            selector_feats[7] = variance;//距离方差
            selector_feats[8] = avg_hop;//平均跳数特征
            selector_feats[9] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
            selector_feats[10] = dist_distribution_entropy;//距离分布熵
            selector_feats[11] = cs.mean;
            selector_feats[12] = cs.variance;
            selector_feats[13] = cs.direction_entropy;


            int selector_num_rows = 1;
            int selector_num_feats = selector_total_feats;
//        double predictor_start_time = data_manager.elapsed_secs();

//        double predictor_end_time = data_manager.elapsed_secs();
            double out_result_selector[1];
            long int out_len_selector;
            int res_selector = LGBM_BoosterPredictForMatSingleRow(
                    booster_selector,
                    selector_feats.data(),
                    C_API_DTYPE_FLOAT64,
                    selector_num_feats,
                    1,
                    C_API_PREDICT_NORMAL,
                    0,
                    -1,
                    "",
                    &out_len_selector,
                    out_result_selector);

            if (res_selector != 0) {
                printf("Error predicting recall\n");
                exit(1);
            }
            // probability of class=1
            double prob = out_result_selector[0];
            // threshold = 0.5 (you can tune it)
            int cls = (prob >= 0.5) ? 1 : 0;
            if(cls==0){
                //1使用recall，否则使用CNS
                return -1.0;
            }
        }

        //删除入口点距离特征
        int total_feats = 9;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

//        feats[3] = d_start;//底层入口点
        feats[3] = visited_points_k;//CNS中已经访问的数据点数量,在hnsw中难以实现，使用CNS中当前访问点到查询点距离与最远数据点到查询点距离比值代替
        feats[4] = duration;
        feats[5] = d_1st;//0号位置数据点
        feats[6] = furthest_dist;//k-1号位置数据点

        feats[7] = avg_dist;//距离均值
        feats[8] = variance;//距离方差

        int num_rows = 1;
        int num_feats = total_feats;


        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                //需要更改
                booster_recall,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_recall = (float) out_result[0];
        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

        if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
            predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
        }

//        float recall_diff = target_recall - predicted_recall;
//        *prediction_interval =min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%f,", first_nn_dis);

            fprintf(log_file, "%d,", visited_points);
            fprintf(log_file, "%d,", duration);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_recall);
        }

        return predicted_recall;
    }
    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETModelSelectPredictorHNSW::predict_CNS_test_Noquery_Noske_Nodensty(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 14;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats ;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);

        //todo 根据距离分布熵进行判断是否要进行预测准确率的模型预测，需要进行模型预测，返回模型预测值，否则返回0

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点
        feats[6] = avg_dist;//距离均值
        feats[7] = variance;//距离方差
        feats[8] = avg_hop;//平均跳数特征
//        feats[9] = density;//密度特征特征
        feats[9] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[10] = dist_distribution_entropy;//距离分布熵
//        feats[12] = skewness;
//        feats[13] = kurtosis;
//        feats[14] = energy;
        //密度特征
        feats[11] = cs.mean;
        feats[12] = cs.variance;
        feats[13] = cs.direction_entropy;


        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster_CNS,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

//        double raw_pred = out_result[0];
//        /* 1️⃣ safety floor before scaling */
//        raw_pred = std::max<double>(raw_pred, k);
//        /* 2️⃣ γ 校准（关键） */
//        int predicted_CNS = (int) std::ceil(1.0328 * raw_pred);
//        /* 3️⃣ safety floor & ceiling */
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        // LightGBM 输出的是 log1p(label)
//        double y_pred_log = out_result[0];
//        // 1️⃣ log → 原空间
//        double y_pred = std::expm1(y_pred_log);
//        // 2️⃣ 四舍五入 or 向上取整（efSearch 建议 ceil）
//        int predicted_CNS = static_cast<int>(std::ceil(y_pred));
//        // 3️⃣ 安全裁剪
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        int predicted_CNS = (int) out_result[0];

        //使用区域放大因子进行放大
//        if(predicted_CNS>150 && predicted_CNS<=160){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.017397));
//        }else if(predicted_CNS>160 && predicted_CNS<=170){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.051415));
//        }else if(predicted_CNS>170 && predicted_CNS<=180){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.069137));
//        }else if(predicted_CNS>180 && predicted_CNS<=190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.060578));
//        }else if(predicted_CNS>190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.030164));
//        }

        predicted_CNS = std::max(k, std::min(efSearch, predicted_CNS));  // Make sure recall is in [k, efSearch]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }
        return predicted_CNS;
    }
    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETModelSelectPredictorHNSW::predict_CNS_Model_Select_test_Noquery_Noske_Nodensty(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 14;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats ;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);

        //todo 根据距离分布熵进行判断是否要进行预测准确率的模型预测，需要进行模型预测，返回模型预测值，否则返回0

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点
        feats[6] = avg_dist;//距离均值
        feats[7] = variance;//距离方差
        feats[8] = avg_hop;//平均跳数特征
//        feats[9] = density;//密度特征特征
        feats[9] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[10] = dist_distribution_entropy;//距离分布熵
//        feats[12] = skewness;
//        feats[13] = kurtosis;
//        feats[14] = energy;
        //密度特征
        feats[11] = cs.mean;
        feats[12] = cs.variance;
        feats[13] = cs.direction_entropy;


        int num_rows = 1;
        int num_feats = total_feats;
//        double predictor_start_time = data_manager.elapsed_secs();

//        double predictor_end_time = data_manager.elapsed_secs();
        double out_result_selector[1];
        long int out_len_selector;
        int res_selector = LGBM_BoosterPredictForMatSingleRow(
                booster_selector,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len_selector,
                out_result_selector);

        if (res_selector != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }
        // probability of class=1
        double prob = out_result_selector[0];
        // threshold = 0.5 (you can tune it)
        int cls = (prob >= 0.5) ? 1 : 0;
        if(cls==1){
            //1使用recall，否则使用CNS
            return -1.0;
        }


        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster_CNS,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

//        double raw_pred = out_result[0];
//        /* 1️⃣ safety floor before scaling */
//        raw_pred = std::max<double>(raw_pred, k);
//        /* 2️⃣ γ 校准（关键） */
//        int predicted_CNS = (int) std::ceil(1.0328 * raw_pred);
//        /* 3️⃣ safety floor & ceiling */
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        // LightGBM 输出的是 log1p(label)
//        double y_pred_log = out_result[0];
//        // 1️⃣ log → 原空间
//        double y_pred = std::expm1(y_pred_log);
//        // 2️⃣ 四舍五入 or 向上取整（efSearch 建议 ceil）
//        int predicted_CNS = static_cast<int>(std::ceil(y_pred));
//        // 3️⃣ 安全裁剪
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        int predicted_CNS = (int) out_result[0];

        //使用区域放大因子进行放大
//        if(predicted_CNS>150 && predicted_CNS<=160){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.017397));
//        }else if(predicted_CNS>160 && predicted_CNS<=170){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.051415));
//        }else if(predicted_CNS>170 && predicted_CNS<=180){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.069137));
//        }else if(predicted_CNS>180 && predicted_CNS<=190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.060578));
//        }else if(predicted_CNS>190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.030164));
//        }

        predicted_CNS = std::max(k, std::min(efSearch, predicted_CNS));  // Make sure recall is in [k, efSearch]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }
        return predicted_CNS;
    }
    //核心方法，这个函数基于当前 HNSW 搜索状态预测 CNS，用于判断是否继续搜索。
    float RAETModelSelectPredictorHNSW::predict_CNS_Select_test_Noquery_Noske_Nodensty(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,//已插入到候选堆的数量
            double elapsed,
            float first_nn_dis,
            int k,
//            int *prediction_interval,
            double *predictor_time,
            int efSearch,
            float entry_query_dist_ratio,
            float best_dist_so_far) {
        if (total_insertions < data_manager.k) {
            return 0.0;
        }

        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);

        int num_nonquery_feats = 14;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats ;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
        float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
        float variance = data_manager.get_variance_of_query(query_idx);

        //平均跳数特征
        float avg_hop = data_manager.get_avg_hop_of_query(query_idx);
        //密度特征，即是平均距离的倒数
        float density = 1.0/avg_dist;
        //距离分布熵，传入查询点id和CNS中的d0
        float dist_distribution_entropy = data_manager.get_dist_distribution_entropy(query_idx,best_dist_so_far);
        //余弦特征
        CosineStats cs = data_manager.get_cosine_stats_of_CNS(query_idx);

        //三个统计特征，Skewness（偏度），Kurtosis（峰度），Energy
        float skewness = data_manager.get_skewness_of_query(query_idx);
        float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
        float energy = data_manager.get_energy_of_query(query_idx);

        //todo 根据距离分布熵进行判断是否要进行预测准确率的模型预测，需要进行模型预测，返回模型预测值，否则返回0

        feats[0] = nstep;//跳数
        feats[1] = ndis;//
        feats[2] = total_insertions;//插入到CNS前k位置数据点数量

        feats[3] = d_start;//底层入口点
        feats[4] = d_1st;//0号位置数据点
        feats[5] = furthest_dist;//k-1号位置数据点
        feats[6] = avg_dist;//距离均值
        feats[7] = variance;//距离方差
        feats[8] = avg_hop;//平均跳数特征
//        feats[9] = density;//密度特征特征
        feats[9] = entry_query_dist_ratio;//入口点距离与入口点邻居的比值特征
        feats[10] = dist_distribution_entropy;//距离分布熵
//        feats[12] = skewness;
//        feats[13] = kurtosis;
//        feats[14] = energy;
        //密度特征
        feats[11] = cs.mean;
        feats[12] = cs.variance;
        feats[13] = cs.direction_entropy;
        //SIFT k50 tr90
//        if ((d_1st <= 48975.0 && total_insertions <= 200.5) ||
//            (d_1st <= 48975.0 && total_insertions > 200.5 && cs.mean > 0.9) ||
//            (d_1st > 48975.0 && ndis > 2071.5))
//        {
//            return -1;
//        }
        //GLOVE100 k50 tr90
//        if (furthest_dist <= 22.15 && ndis <= 765.50) {
//            return -1;
//        }
        //deep k50 tr90
//        if (d_start <= 0.46 && (ndis <= 1449.50 || avg_hop > 10.50)) {
//            return -1;
//        }
        int num_feats = total_feats;
        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster_CNS,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

//        double raw_pred = out_result[0];
//        /* 1️⃣ safety floor before scaling */
//        raw_pred = std::max<double>(raw_pred, k);
//        /* 2️⃣ γ 校准（关键） */
//        int predicted_CNS = (int) std::ceil(1.0328 * raw_pred);
//        /* 3️⃣ safety floor & ceiling */
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        // LightGBM 输出的是 log1p(label)
//        double y_pred_log = out_result[0];
//        // 1️⃣ log → 原空间
//        double y_pred = std::expm1(y_pred_log);
//        // 2️⃣ 四舍五入 or 向上取整（efSearch 建议 ceil）
//        int predicted_CNS = static_cast<int>(std::ceil(y_pred));
//        // 3️⃣ 安全裁剪
//        predicted_CNS = std::max(k, predicted_CNS);
//        predicted_CNS = std::min(efSearch, predicted_CNS);

        int predicted_CNS = (int) out_result[0];

        //使用区域放大因子进行放大
//        if(predicted_CNS>150 && predicted_CNS<=160){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.017397));
//        }else if(predicted_CNS>160 && predicted_CNS<=170){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.051415));
//        }else if(predicted_CNS>170 && predicted_CNS<=180){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.069137));
//        }else if(predicted_CNS>180 && predicted_CNS<=190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.060578));
//        }else if(predicted_CNS>190){
//            predicted_CNS=static_cast<int>(std::ceil(predicted_CNS * 1.030164));
//        }

        predicted_CNS = std::max(k, std::min(efSearch, predicted_CNS));  // Make sure recall is in [k, efSearch]

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        // This logging is only for debugging purposes and may be used only when no multithreading is used
        if (log_file && per_prediction_logging) {
            double actual_recall_k = data_manager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);
            fprintf(log_file, "%f,", first_nn_dis);

//            fprintf(log_file, "%d,", visited_points);

            fprintf(log_file, "%f,", d_1st);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);
            fprintf(log_file, "%f,", variance);
//            fprintf(log_file, "%f,", median);
//
//            fprintf(log_file, "%f,", percentile_25);
//            fprintf(log_file, "%f,", percentile_75);

//            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%f,", avg_hop);
            fprintf(log_file, "%f,", density);
            fprintf(log_file, "%f,", entry_query_dist_ratio);
            fprintf(log_file, "%f,", dist_distribution_entropy);
            fprintf(log_file, "%f,", skewness);
            fprintf(log_file, "%f,", kurtosis);
            fprintf(log_file, "%f,", energy);
            fprintf(log_file, "%f,", cs.mean);
            fprintf(log_file, "%f,", cs.variance);
            fprintf(log_file, "%f,", cs.direction_entropy);

            fprintf(log_file, "%d,", 1);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%d\n", predicted_CNS);
        }
        return predicted_CNS;
    }
    void RAETModelSelectPredictorHNSW::log_final_recall_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            double last_predicted_recall,
//            int prediction_interval,
            int query_predictor_calls,
            double predictor_time,
            int visited_points,
            int duration) {
        if (!log_file || per_prediction_logging) {
            return;
        }

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);


        fprintf(log_file, "%f,", first_nn_dis);
        fprintf(log_file, "%d,", visited_points);
        fprintf(log_file, "%d,", duration);

        fprintf(log_file, "%f,", data_manager.get_nearest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_avg_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_furthest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", data_manager.get_variance_of_query(query_idx));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.50));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.25));
//        fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.75));

//        fprintf(log_file, "%d,", prediction_interval);
        fprintf(log_file, "%d,", query_predictor_calls);
        fprintf(log_file, "%f,", predictor_time * 1000);

        fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
        fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
        fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

        fprintf(log_file, "%f,", data_manager.get_recallk(query_idx));
        fprintf(log_file, "%f\n", last_predicted_recall);
    }
    RAETModelQuerySelectPredictorHNSW::RAETModelQuerySelectPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            char *model_selector_path)
            : data_manager(data_manager)
               {//构造函数方法 将参数复制到成员变量
        int out_iterations;
        int result_model_selector = LGBM_BoosterCreateFromModelfile(
                model_selector_path, &out_iterations, &booster_selector);
        if (result_model_selector != 0 ) {
            exit(1);
        }
        LGBM_SetMaxThreads(1);
    }



// LAETPredictorHNSW
    LAETPredictorHNSW::LAETPredictorHNSW(
            DeclarativeRecallDataManager data_manager,
            int fixed_amount_of_distance_calcs,
            float prediction_multiplier,
            char *predictor_model_path)
            : data_manager(data_manager),
              fixed_amount_of_distance_calcs(fixed_amount_of_distance_calcs),
              prediction_multiplier(prediction_multiplier) {
        int out_iterations;
        int result = LGBM_BoosterCreateFromModelfile(predictor_model_path, &out_iterations, &booster);
        if (result != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }

    void LAETPredictorHNSW::init_log_file() {
        if (!data_manager.log_filename) {
            return;
        }

        log_file = fopen(data_manager.log_filename, "w");
        if (!log_file) {
            // printf("Error opening recall data log file\n");
            perror("Error opening recall data log file");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");

        fprintf(log_file, "predictor_time,");
        fprintf(log_file, "predicted_distance_calcs,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r\n");
    }

    void LAETPredictorHNSW::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

    int LAETPredictorHNSW::predict_distance_calcs(
            idx_t query_idx,
            int nstep,
            int ndis,
            double elapsed,
            float first_nn_dis,
            double *predictor_time) {
        double predictor_start_time = data_manager.elapsed_secs();

        float d_start = first_nn_dis;
        float d_1st = data_manager.get_nearest_dist_of_query(query_idx);
        float d_10th = -1;
        if (data_manager.k >= 10) {
            d_10th = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
        }

        float d_1st_to_start = -1;
        if (d_start > 0) {
            d_1st_to_start = d_1st / d_start;
        }

        float d_10th_to_start = -1;
        if (d_10th != -1 && d_start > 0) {
            d_10th_to_start = d_10th / d_start;
        }

        int num_nonquery_feats = 5;
        int query_feats = data_manager.d;
        int total_feats = num_nonquery_feats + query_feats;
        std::vector<double> feats(static_cast<size_t>(total_feats));

        feats[0] = d_start;
        feats[1] = d_1st;
        feats[2] = d_10th;
        feats[3] = d_1st_to_start;
        feats[4] = d_10th_to_start;

        for (int i = 0; i < data_manager.d; i++) {
            feats[static_cast<size_t>(num_nonquery_feats + i)] =
                    data_manager.queries[query_idx * data_manager.d + i];
        }

        int num_rows = 1;
        int num_feats = total_feats;

        double out_result[1];
        long int out_len;
        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                feats.data(),
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting distance calcs\n");
            exit(1);
        }

        int predicted_distance_calcs = (int) out_result[0];

        predicted_distance_calcs = static_cast<int>(std::round(prediction_multiplier * predicted_distance_calcs));

        double predictor_end_time = data_manager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        if (per_prediction_logging && log_file) {
            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%f,", *predictor_time * 1000);
            fprintf(log_file, "%d,", predicted_distance_calcs);
            fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
            fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
            fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));
            fprintf(log_file, "%f\n", data_manager.get_recallk(query_idx));
        }

        return predicted_distance_calcs;
    }

    void LAETPredictorHNSW::log_final_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            double elapsed,
            float first_nn_dis,
            int predicted_distance_calcs,
            double predictor_time) {
        if (!log_file || per_prediction_logging) {
            return;
        }

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%f,", predictor_time * 1000);
        fprintf(log_file, "%d,", predicted_distance_calcs);
        fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
        fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
        fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));
        fprintf(log_file, "%f\n", data_manager.get_recallk(query_idx));
    }

// DARTHPredictorIVF
    DARTHPredictorIVF::DARTHPredictorIVF(
            DeclarativeRecallDataManager dataManager,
            double target_recall,
            int initial_prediction_interval,
            int min_prediction_interval,
            bool per_prediction_logging,
            char *predictor_model_path,
            int logging_interval)
            : dataManager(dataManager),
              target_recall(target_recall),
              initial_prediction_interval(initial_prediction_interval),
              min_prediction_interval(min_prediction_interval),
              per_prediction_logging(per_prediction_logging),
              logging_interval(logging_interval) {
        int out_iterations;
        int result = LGBM_BoosterCreateFromModelfile(
                predictor_model_path, &out_iterations, &booster);
        if (result != 0) {
            exit(1);
        }

        LGBM_SetMaxThreads(1);
    }

    void DARTHPredictorIVF::init_log_file() {
        if (!dataManager.log_filename) {
            return;
        }

        log_file = fopen(dataManager.log_filename, "w");
        if (!log_file) {
            printf("Error opening recall data log file\n");
            exit(1);
        }

        fprintf(log_file, "qid,");
        fprintf(log_file, "step,");
        fprintf(log_file, "dists,");
        fprintf(log_file, "elaps_ms,");
        fprintf(log_file, "inserts,");

        fprintf(log_file, "first_nn_dist,");

        fprintf(log_file, "nn_dist,");
        fprintf(log_file, "avg_dist,");
        fprintf(log_file, "furthest_dist,");

        fprintf(log_file, "variance,");
        fprintf(log_file, "median,");

        fprintf(log_file, "percentile_25,");
        fprintf(log_file, "percentile_75,");

        fprintf(log_file, "r_current_interval,");
        fprintf(log_file, "r_predictor_calls,");
        fprintf(log_file, "r_predictor_time_ms,");

        // Quality approximation measures
        fprintf(log_file, "RDE,");  // relative distance error
        fprintf(log_file, "TDR,");  // total distance ratio
        fprintf(log_file, "NRS,");  // normalized rank sum

        fprintf(log_file, "r_actual,");
        fprintf(log_file, "r_predicted\n");
    }

    void DARTHPredictorIVF::close_log_file() {
        if (log_file) {
            fclose(log_file);
        }
    }

    float DARTHPredictorIVF::predictRecall(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            int query_predictor_calls,
            int *prediction_interval,
            double *predictor_time) {
        if (total_insertions < dataManager.k) {
            return 0.0f;
        }

        query_predictor_calls++;

        double predictor_start_time = dataManager.elapsed_secs();

        float nn_dist = dataManager.get_nearest_dist_of_query(query_idx);
        float furthest_dist = dataManager.get_furthest_dist_of_query(query_idx);
        float avg_dist = dataManager.get_avg_dist_of_query(query_idx);
        float variance = dataManager.get_variance_of_query(query_idx);
        float median = dataManager.get_percentile_of_query(query_idx, 0.50);
        float percentile_25 = dataManager.get_percentile_of_query(query_idx, 0.25);
        float percentile_75 = dataManager.get_percentile_of_query(query_idx, 0.75);

        const int num_feats = 11;
        const double data[11] = {
                (double) nstep,
                (double) ndis,
                (double) total_insertions,
                (double) first_nn_dis,
                nn_dist,
                avg_dist,
                furthest_dist,
                variance,
                median,
                percentile_25,
                percentile_75};

        double out_result[1];
        long int out_len;

        int res = LGBM_BoosterPredictForMatSingleRow(
                booster,
                data,
                C_API_DTYPE_FLOAT64,
                num_feats,
                1,
                C_API_PREDICT_NORMAL,
                0,
                -1,
                "",
                &out_len,
                out_result);

        if (res != 0) {
            printf("Error predicting recall\n");
            exit(1);
        }

        float predicted_recall = (float) out_result[0];
        predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

        if (dataManager.k == 1) {  // Special case for k=1 where recall is 0 or 1
            predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
        }

        float recall_diff = static_cast<float>(target_recall) - predicted_recall;
        *prediction_interval = min_prediction_interval +
                               (initial_prediction_interval - min_prediction_interval) *
                               recall_diff;

        *prediction_interval = ((*prediction_interval + logging_interval - 1) / logging_interval) * logging_interval;

        double predictor_end_time = dataManager.elapsed_secs();
        *predictor_time += predictor_end_time - predictor_start_time;

        if (log_file && per_prediction_logging) {
            double actual_recall_k = dataManager.get_recallk(query_idx);

            fprintf(log_file, "%ld,", query_idx);
            fprintf(log_file, "%d,", nstep);
            fprintf(log_file, "%d,", ndis);
            fprintf(log_file, "%f,", elapsed * 1000);
            fprintf(log_file, "%d,", total_insertions);

            fprintf(log_file, "%f,", first_nn_dis);
            fprintf(log_file, "%f,", nn_dist);
            fprintf(log_file, "%f,", avg_dist);
            fprintf(log_file, "%f,", furthest_dist);

            fprintf(log_file, "%f,", variance);
            fprintf(log_file, "%f,", median);

            fprintf(log_file, "%f,", percentile_25);
            fprintf(log_file, "%f,", percentile_75);

            fprintf(log_file, "%d,", *prediction_interval);
            fprintf(log_file, "%d,", query_predictor_calls);
            fprintf(log_file, "%f,", *predictor_time * 1000);

            fprintf(log_file, "%f,", dataManager.get_RDE(query_idx));
            fprintf(log_file, "%f,", dataManager.get_TDR(query_idx));
            fprintf(log_file, "%f,", dataManager.get_NRS(query_idx));

            fprintf(log_file, "%f,", actual_recall_k);
            fprintf(log_file, "%f\n", predicted_recall);
        }

        return predicted_recall;
    }

    void DARTHPredictorIVF::log_final_recall_result(
            idx_t query_idx,
            int nstep,
            int ndis,
            int total_insertions,
            double elapsed,
            float first_nn_dis,
            double last_predicted_recall,
            int prediction_interval,
            int query_predictor_calls,
            double predictor_time) {
        if (!log_file || per_prediction_logging) {
            return;
        }

        fprintf(log_file, "%ld,", query_idx);
        fprintf(log_file, "%d,", nstep);
        fprintf(log_file, "%d,", ndis);
        fprintf(log_file, "%f,", elapsed * 1000);
        fprintf(log_file, "%d,", total_insertions);

        fprintf(log_file, "%f,", first_nn_dis);

        fprintf(log_file, "%f,", dataManager.get_nearest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", dataManager.get_avg_dist_of_query(query_idx));
        fprintf(log_file, "%f,", dataManager.get_furthest_dist_of_query(query_idx));
        fprintf(log_file, "%f,", dataManager.get_variance_of_query(query_idx));
        fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.50));
        fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.25));
        fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.75));

        fprintf(log_file, "%d,", prediction_interval);
        fprintf(log_file, "%d,", query_predictor_calls);
        fprintf(log_file, "%f,", predictor_time * 1000);

        fprintf(log_file, "%f,", dataManager.get_RDE(query_idx));
        fprintf(log_file, "%f,", dataManager.get_TDR(query_idx));
        fprintf(log_file, "%f,", dataManager.get_NRS(query_idx));

        fprintf(log_file, "%f,", dataManager.get_recallk(query_idx));
        fprintf(log_file, "%f\n", last_predicted_recall);
    }


}  // namespace faiss
