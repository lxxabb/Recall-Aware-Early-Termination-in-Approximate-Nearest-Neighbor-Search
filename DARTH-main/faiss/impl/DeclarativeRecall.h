#pragma once

#include <LightGBM/boosting.h>
#include <LightGBM/c_api.h>
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <sys/time.h>

#include <string>
#include "AuxIndexStructures.h"

#define INTERVAL_DISABLED_VALUE -1
#define INTERVAL_WHEN_RECALL_UPDATES_VALUE 0

namespace faiss {
    struct CosineStats {
        float mean;              // 余弦均值
        float variance;          // 余弦方差
        float direction_entropy; // 方向熵
    };

    typedef struct DeclarativeRecallDataManager {
        idx_t nq, ndb, k, d;

        const float *queries = nullptr;
        const float *db = nullptr;

        char *log_filename = nullptr;

        float *distances = nullptr;
        idx_t *labels = nullptr;

        idx_t *gt = nullptr;
        float *gt_dist = nullptr;

        // including all possible k-nn neighbors and their size (needed for normalized rank sum)包括所有可能的 k 近邻及其大小（归一化秩和所必需）
        idx_t *gt_for_all_k = nullptr;
        idx_t k_all = 0;

        // 🔑 新增：出度指针（必须）
        const std::vector<int>* out_degree_L0 = nullptr;
        //🔑 新增 每个数据点的跳数===== dynamic per-query hop (timestamped) =====
        std::vector<int>* node_hop = nullptr;
        std::vector<int>* node_hop_qid = nullptr;
//        // ===== 新增：CNS diff gating =====
//        std::vector<int> last_pred_mark;
//        int pred_epoch = 0;

        // ⭐⭐⭐ 新增两个成员变量（必须有，否则初始化报错）
        const float* norm_queries = nullptr;
        const float* norm_db = nullptr;

        //预归一化，方便余弦相似度计算
        bool normalized = false;
        void normalize_all_vectors();


        DeclarativeRecallDataManager();

        DeclarativeRecallDataManager(
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
                const std::vector<int>* out_degree_L0,
                std::vector<int>* node_hop,
                std::vector<int>* node_hop_qid,
                // ⭐⭐ 新增两个指针归一化数据
                const float* norm_queries,
                const float* norm_db);

        int get_out_degree(idx_t node) const;
        void record_hop(idx_t node, int hop, idx_t q_id);

        //归一化代码
        void l2_normalize(float* x, size_t d) {
            float norm = 0.0f;
            for (size_t i = 0; i < d; i++)
                norm += x[i] * x[i];
            norm = std::sqrt(norm) + 1e-12f;

            for (size_t i = 0; i < d; i++)
                x[i] /= norm;
        }

//        int  get_hop(idx_t node, idx_t query_idx) const;

//        // ===== Predictor gating  =====
//        void snapshot_labels(idx_t q_id);
//        int count_new_labels_since_last_snapshot(idx_t q_id) const;
        //欧式距离
        float get_ed(const float *x, const float *y, size_t d);
        //余弦相似度
        float cosine_similarity(const float *x, const float *y, size_t d);

        double elapsed_secs();

        float get_avg_dist_of_query(idx_t q_id);
        float get_avg_degree_of_query(idx_t q_id);
        float get_avg_hop_of_query(idx_t q_id);
        float get_dist_distribution_entropy(idx_t q_id,float d0);

        float get_nearest_dist_of_query(idx_t q_id);
        int get_visited_points_k_query(idx_t q_id,VisitedTable& vt);
        idx_t get_nearest_id_of_query(idx_t q_id);

        float get_furthest_dist_of_query(idx_t q_id);

        float get_dist_of_query_to_medoid(idx_t q_id);

        CosineStats get_cosine_stats_of_CNS(idx_t q_id);

        float get_recallk(idx_t query_idx);

        float get_kth_nearest_dist_of_query(idx_t q_id, int kth);

        float get_variance_of_query(idx_t q_id);

        float get_percentile_of_query(idx_t q_id, float percentile);

        float get_skewness_of_query(idx_t q_id);

        float get_energy_of_query(idx_t q_id);

        float get_kurtosis_of_query(idx_t q_id);

        float get_TDR(idx_t q_id);

        float get_RDE(idx_t q_id);

        float get_NRS(idx_t q_id);

    } DeclarativeRecallDataManager;

    typedef struct DeclarativeRecallDataCollectorHNSW {
        DeclarativeRecallDataManager data_manager;
        FILE *log_file = nullptr;

        int logging_interval = 1;

        bool do_naive_early_stop = false;
        int dist_early_stop_threshold = 1000;

        bool include_data_dimensions = true;

        DeclarativeRecallDataCollectorHNSW();

        DeclarativeRecallDataCollectorHNSW(
                DeclarativeRecallDataManager data_manager,
                int logging_interval = INTERVAL_DISABLED_VALUE,
                int dist_early_stop_threshold = 0);

        void init_log_file();
        void init_CNS_log_file();

        void close_log_file();

        void append_to_log(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k,
                int visited_points);

        std::string get_observation_data_str(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k,
                int visited_points,
                int duration);
        std::string get_observation_CNS_data_str(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k,
                int visited_points,
                float entry_query_dist_ratio,
                float best_dist_so_far);

        void flush_observation_to_log(std::string observation_data);

        void flush_all_observations_to_log(std::string *observations, int n);

    } DeclarativeRecallDataCollectorHNSW;

    typedef struct DARTHPredictorHNSW {
        //收集搜索过程中的特征数据（nstep、distance 统计、插入数量等）。
        //预测器依赖它来构建 LightGBM 模型输入。
        DeclarativeRecallDataManager data_manager;

        //日志文件指针，用于记录预测过程、最终召回结果等。
        FILE *log_file = nullptr;
        //是否对每次预测都写一条日志。
        bool per_prediction_logging = false;

        double target_recall;

        int initial_prediction_interval = 2000, min_prediction_interval = 100;

        BoosterHandle booster;

        DARTHPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                double target_recall,
                int initial_prediction_interval,
                int min_prediction_interval,
                bool per_prediction_logging,
                char *predictor_model_path);

        void init_log_file();

        void close_log_file();

        //输入是 HNSW 搜索的中间状态：
        float predict_recall(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                int *prediction_interval,
                double *predictor_time);

        //在一个 query 结束时写最终日志：
        void log_final_recall_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                double last_predicted_recall,
                int prediction_interval,
                int query_predictor_calls,
                double predictor_time);

    } DARTHPredictorHNSW;

    typedef struct RAETPredictorHNSW {
        //收集搜索过程中的特征数据（nstep、distance 统计、插入数量等）。
        //预测器依赖它来构建 LightGBM 模型输入。
        DeclarativeRecallDataManager data_manager;

        //日志文件指针，用于记录预测过程、最终召回结果等。
        FILE *log_file = nullptr;
        //是否对每次预测都写一条日志。
        bool per_prediction_logging = false;

        double target_recall;

        int stability_times = 20;
        int stability_times_r1 = 20;

        BoosterHandle booster;

        RAETPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                double target_recall,
                int stability_times,
                int stability_times_r1,
                bool per_prediction_logging,
                char *predictor_model_path);

        void init_log_file();

        void close_log_file();

        //输入是 HNSW 搜索的中间状态：
        float predict_recall(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                double *predictor_time,
                int visited_points,
                int duration);

        //在一个 query 结束时写最终日志：
        void log_final_recall_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                double last_predicted_recall,
                int query_predictor_calls,
                double predictor_time,
                int visited_points,
                int duration);

    } RAETPredictorHNSW;

    typedef struct RAETCNSPredictorHNSW {
        //收集搜索过程中的特征数据（nstep、distance 统计、插入数量等）。
        //预测器依赖它来构建 LightGBM 模型输入。
        DeclarativeRecallDataManager data_manager;

        //日志文件指针，用于记录预测过程、最终召回结果等。
        FILE *log_file = nullptr;
        //是否对每次预测都写一条日志。
        bool per_prediction_logging = false;
        bool selector=false;
        double target_recall;

        BoosterHandle booster;
        BoosterHandle CNS_booster;

        RAETCNSPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                double target_recall,
                bool per_prediction_logging,
                char *predictor_model_path);
        RAETCNSPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                double target_recall,
                bool per_prediction_logging,
                char *predictor_model_path,
                bool selector);

        void init_log_file();

        void close_log_file();

        //输入是 HNSW 搜索的中间状态：
        float predict_CNS(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                double *predictor_time,
                int visited_points);
        //回归模型
        float predict_CNS_test(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);
        float predict_CNS_test_Noquery_Noske_Nodensty(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);
        //二分类模型
        float predict_CNS_test_classification(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);
        //二分类 + 回归模型
        float predict_CNS_test_classification_regression(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);

        //在一个 query 结束时写最终日志：
        void log_final_recall_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
//                double last_predicted_recall,
                int query_predictor_calls,
                double predictor_time,
                float best_dist_so_far,
                float ratio,
                int predict_CNS);

    } RAETCNSPredictorHNSW;
    typedef struct RAETModelSelectPredictorHNSW {
        //收集搜索过程中的特征数据（nstep、distance 统计、插入数量等）。
        //预测器依赖它来构建 LightGBM 模型输入。
        DeclarativeRecallDataManager data_manager;

        //日志文件指针，用于记录预测过程、最终召回结果等。
        FILE *log_file = nullptr;
        //是否对每次预测都写一条日志。
        bool per_prediction_logging = false;

        double target_recall;

        int stability_times = 20;
        int stability_times_r1 = 20;

        BoosterHandle booster_recall;
        BoosterHandle booster_CNS;
        BoosterHandle booster_selector;

        RAETModelSelectPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                double target_recall,
                int stability_times,
                int stability_times_r1,
                bool per_prediction_logging,
                char *predictor_model_path_recall,
                char *predictor_model_path_CNS,
                char *model_selector_path);

        void init_log_file();
        void init_Select_Model_log_file();

        void close_log_file();

        //输入是 HNSW 搜索的中间状态：
        float predict_recall(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                double *predictor_time,
                int visited_points,
                int duration,
                float best_dist_so_far);
        float Model_Select_predict_recall(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                double *predictor_time,
                int visited_points,
                int duration,
                float best_dist_so_far,
                float ratio);

        float predict_CNS_test_Noquery_Noske_Nodensty(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);
        float predict_CNS_Model_Select_test_Noquery_Noske_Nodensty(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);
        float predict_CNS_Select_test_Noquery_Noske_Nodensty(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int k,
                double *predictor_time,
                int efSearch,
                float ratio,
                float best_dist_so_far);

        //在一个 query 结束时写最终日志：
        void log_final_recall_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                double last_predicted_recall,
                int query_predictor_calls,
                double predictor_time,
                int visited_points,
                int duration);
        std::string get_observation_CNS_data_str(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k,
                int visited_points,
                float entry_query_dist_ratio,
                float best_dist_so_far);

        void flush_observation_to_log(std::string observation_data);

        void flush_all_observations_to_log(std::string *observations, int n);

    } RAETModelSelectPredictorHNSW;
    typedef struct RAETModelQuerySelectPredictorHNSW {
        DeclarativeRecallDataManager data_manager;
        //是否对每次预测都写一条日志。
        BoosterHandle booster_selector;

        RAETModelQuerySelectPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                char *model_selector_path);
    } RAETModelQuerySelectPredictorHNSW;

    typedef struct LAETPredictorHNSW {
        DeclarativeRecallDataManager data_manager;
        int fixed_amount_of_distance_calcs = 100;
        float prediction_multiplier = 1;
        FILE *log_file = nullptr;
        bool per_prediction_logging = false;

        BoosterHandle booster;

        LAETPredictorHNSW(
                DeclarativeRecallDataManager data_manager,
                int fixed_amount_of_distance_calcs,
                float prediction_multiplier,
                char *predictor_model_path);

        void init_log_file();

        void close_log_file();

        int predict_distance_calcs(
                idx_t query_idx,
                int nstep,
                int ndis,
                double elapsed,
                float first_nn_dis,
                double *predictor_time);

        void log_final_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                double elapsed,
                float first_nn_dis,
                int predicted_distance_calcs,
                double predictor_time);

    } LAETPredictorHNSW;

    typedef struct DeclarativeRecallDataCollectorIVF {
        // ongoing: adding multithreading support

        DeclarativeRecallDataManager dataManager;
        FILE *log_file = nullptr;

        int logging_interval = 1;

        bool do_naive_early_stop = false;
        int dist_early_stop_threshold = 1000;

        bool include_data_dimensions = true;

        DeclarativeRecallDataCollectorIVF() {}

        DeclarativeRecallDataCollectorIVF(
                DeclarativeRecallDataManager dataManager,
                int logging_interval,  //= INTERVAL_DISABLED_VALUE,
                int dist_early_stop_threshold = 0)
                : dataManager(dataManager),
                  logging_interval(logging_interval),
                  dist_early_stop_threshold(dist_early_stop_threshold) {
            if (dist_early_stop_threshold > 0) {
                do_naive_early_stop = true;
            }
        }

        void init_log_file();

        void close_log_file();

        void append_to_log(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k);

        /*
        std::string get_observation_data_str(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                float recall_k) {
            std::string observation_data = "";

            if (!log_file || total_insertions < dataManager.k) {
                return observation_data;
            }

            double feature_collection_time_start = dataManager.elapsed_secs();

            observation_data += std::to_string(query_idx) + ",";
            observation_data += std::to_string(nstep) + ",";
            observation_data += std::to_string(ndis) + ",";
            observation_data += std::to_string(elapsed * 1000) + ",";
            observation_data += std::to_string(total_insertions) + ",";
            observation_data += std::to_string(first_nn_dis) + ",";

            float nn_dist = dataManager.get_nearest_dist_of_query(query_idx);
            float avg_dist = dataManager.get_avg_dist_of_query(query_idx);
            float furthest_dist = dataManager.get_furthest_dist_of_query(query_idx);
            observation_data += std::to_string(nn_dist) + ",";
            observation_data += std::to_string(avg_dist) + ",";
            observation_data += std::to_string(furthest_dist) + ",";

            float perc_25 = dataManager.get_percentile_of_query(query_idx, 0.25);
            float perc_50 = dataManager.get_percentile_of_query(query_idx, 0.50);
            float perc_75 = dataManager.get_percentile_of_query(query_idx, 0.75);
            float perc_95 = dataManager.get_percentile_of_query(query_idx, 0.95);
            observation_data += std::to_string(perc_25) + ",";
            observation_data += std::to_string(perc_50) + ",";
            observation_data += std::to_string(perc_75) + ",";
            observation_data += std::to_string(perc_95) + ",";

            float variance = dataManager.get_variance_of_query(query_idx);
            observation_data += std::to_string(variance) + ",";

            // New includes start
            float std = std::sqrt(variance);
            float range = furthest_dist - nn_dist;
            float skewness = dataManager.get_skewness_of_query(query_idx);
            float kurtosis = dataManager.get_kurtosis_of_query(query_idx);
            float energy = dataManager.get_energy_of_query(query_idx);
            observation_data += std::to_string(std) + ",";
            observation_data += std::to_string(range) + ",";
            observation_data += std::to_string(skewness) + ",";
            observation_data += std::to_string(kurtosis) + ",";
            observation_data += std::to_string(energy) + ",";
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

            observation_data += std::to_string(dist_10) + ",";
            observation_data += std::to_string(dnn_to_dstart) + ",";
            observation_data += std::to_string(d10_to_dstart) + ",";

            // add the quality approximation measures
            float RDE = dataManager.get_RDE(query_idx);
            float TDR = dataManager.get_TDR(query_idx);
            float NRS = dataManager.get_NRS(query_idx);
            observation_data += std::to_string(RDE) + ",";
            observation_data += std::to_string(TDR) + ",";
            observation_data += std::to_string(NRS) + ",";

            //if (include_data_dimensions){
            //    for (int i = 0; i < dataManager.d; i++) {
            //        observation_data += std::to_string(dataManager.queries[query_idx * dataManager.d + i]) + ",";
            //    }
            //}

            float query_dim_stats[dataManager.summary_stats_num];
            dataManager.get_precomputed_query_stats(query_idx, query_dim_stats);
            for (int i = 0; i < dataManager.summary_stats_num; i++) {
                observation_data += std::to_string(query_dim_stats[i]) + ",";
            }

            double feature_collection_time_end = dataManager.elapsed_secs();
            double feature_collection_time =
                    (feature_collection_time_end - feature_collection_time_start) *
                    1000;
            observation_data += std::to_string(feature_collection_time) + ",";

            observation_data += std::to_string(recall_k) + "\n";

            return observation_data;
        }

        void flush_observation_to_log(std::string observation_data) {
            if (!log_file) {
                return;
            }

            fprintf(log_file, "%s", observation_data.c_str());
        }

        void flush_all_observations_to_log(std::string* observations, int n) {
            if (!log_file) {
                return;
            }

            for (int i = 0; i < n; i++) {
                fprintf(log_file, "%s", observations[i].c_str());
            }
        }
        */
    } DeclarativeRecallDataCollectorIVF;

    typedef struct DARTHPredictorIVF {
        DeclarativeRecallDataManager dataManager;

        int logging_interval = 1;

        FILE *log_file = nullptr;
        bool per_prediction_logging = false;

        double target_recall;

        int initial_prediction_interval = 2000, min_prediction_interval = 100;

        BoosterHandle booster;  // Predictor model

        DARTHPredictorIVF(
                DeclarativeRecallDataManager dataManager,
                double target_recall,
                int initial_prediction_interval,
                int min_prediction_interval,
                bool per_prediction_logging,
                char *predictor_model_path,
                int logging_interval);

        void init_log_file();

        void close_log_file();

        void prepare_for_next_query();

        float predictRecall(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                int query_predictor_calls,
                int *prediction_interval,
                double *predictor_time);

        void log_final_recall_result(
                idx_t query_idx,
                int nstep,
                int ndis,
                int total_insertions,
                double elapsed,
                float first_nn_dis,
                double last_predicted_recall,
                int prediction_interval,
                int query_predictor_calls,
                double predictor_time);

    } DARTHPredictorIVF;

}  // namespace faiss
