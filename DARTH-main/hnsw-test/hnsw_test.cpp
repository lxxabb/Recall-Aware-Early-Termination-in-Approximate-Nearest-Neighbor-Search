#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ComUtil.h"
#include "VectorDataLoader.h"

typedef enum {
    OBSERVATION_DATA_GENERATION = 0,  //为机器学习模型收集训练数据
    DARTH_TESTING = 1,                //使用 DARTH 预测器实现提前终止
    NO_EARLY_TERMINATION_TESTING = 2, //不提前终止：纯 baseline 搜索
    BASELINE_TESTING = 3,             //固定迭代次数停止
    OPTIMAL_RESULTS_GENERATION = 4,   //最优终止点搜索（给 LAET 提供 Ground Truth）
    LAET_TESTING = 5,                  //使用 LAET 预测提前终止
    RAET_TESTING = 6,                //使用 我们的预测准确率模型 提前终止
    RAET_GET_STABILITY_TIMES=7,     //使用预测准确率，得到稳定次数参数
    RAET_CNS_DATA_GENERATION=8,     //预测CNS算法，训练集生成。只收集搜索到K位置时的特征，每个查询点一条特征
    OPTIMAL_EFSEARCH_PER_QUERY = 9,  //预测CNS算法，得到每个查询点达到要求准确率的最小CNS，即标签
    RAET_CNS_TESTING=10,
    RAET_SELECT_MODEL=11,
    RAET_SELECT_MODEL_DATA_GEN=12,
    RAET_QUERY_SELECT_MODEL=13,
} running_mode_t;

char running_mode_str[14][50] = {"Declarative Recall Training Data Generation",
                                "DARTH Testing",
                                "No Early Termination Testing",
                                "Baseline Testing",
                                "Optimal Results Generation",
                                "LAET Testing",
                                "our_RAET Testing",
                                "our_RAET Get Stability Times",
                                "our_RAET Cns Data Generation",
                                "Optimal efSearch per query",
                                "our_RAET CNS Testing",
                                "raet_select_model",
                                "raet_select_model_data_gen",
                                "raet_query_select_model"};

int main(int argc, char **argv) {
    double t0 = elapsed();

    char *dataset, *output = NULL, *index_filepath = NULL, *predictor_model_path = NULL,*predictor_model_path_select_CNS = NULL,*model_selector_path = NULL;
    std::string stability_log_path;
    float target_recall = 0.90;
    int initial_prediction_interval = 1000, min_prediction_interval = 100;

    int logging_interval = 10;

    bool save_index = false;
    running_mode_t mode;
    faiss::idx_t nQ = 100, k = 100;

    int M = 16, efConstruction = 500, efSearch = 500;

    int fixed_amount_of_search = 200;
    float prediction_multiplier = 1.0;

    bool per_prediction_logging = false;
    bool verbose = false;

    int stability_times=20;
    int stability_times_r1=100;
    int train_CNS=500;

    char *dataset_dir_prefix = NULL;

    query_type_t query_type;

    char *noise_perc = "5";

    while (1) {
        static struct option long_options[] = {
                {"dataset",                     required_argument, 0, '0'},
                {"query-num",                   required_argument, 0, '1'},
                {"k",                           required_argument, 0, '2'},
                {"output",                      required_argument, 0, '3'},
                {"M",                           required_argument, 0, '4'},
                {"efConstruction",              required_argument, 0, '5'},
                {"efSearch",                    required_argument, 0, '6'},
                {"index-filepath",              required_argument, 0, '7'},
                {"save-index",                  no_argument,       0, '8'},
                {"mode",                        required_argument, 0, '9'},
                {"target-recall",               required_argument, 0, 'a'},
                {"initial-prediction-interval", required_argument, 0, 'b'},
                {"predictor-model-path",        required_argument, 0, 'c'},
                {"logging-interval",            required_argument, 0, 'd'},
                {"fixed-amount-of-search",      required_argument, 0, 'e'},
                {"prediction-multiplier",       required_argument, 0, 'f'},
                {"min-prediction-interval",     required_argument, 0, 'g'},
                {"per-prediction-logging",      no_argument,       0, 'h'},
                {"verbose",                     no_argument,       0, 'i'},
                {"query-type",                  required_argument, 0, 'j'},
                {"dataset-dir-prefix",          required_argument, 0, 'k'},
                {"gnoise-perc",                 required_argument, 0, 'l'},
                {"help",                        no_argument,       0, 'm'},
                {"stability-times",             required_argument, 0, 'n'},
                {"train-CNS",                   required_argument, 0, 'o'},
                {"stability_log_path",          required_argument, 0, 'p'},
                {"stability-times-r1",          required_argument, 0, 'q'},
                {"predictor-model-path-select-CNS",required_argument, 0, 'r'},
                {"model-selector-path",         required_argument, 0, 's'},
                {NULL, 0, NULL,                                       0}};

        int option_index = 0;
//    使用 getopt_long() 解析命令行参数
        int c = getopt_long(argc, argv, "", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            case '0':
                dataset = optarg;
                break;
            case '1':
                sscanf(optarg, "%ld", &nQ);
                break;
            case '2':
                sscanf(optarg, "%ld", &k);
                break;
            case '3':
                output = optarg;
                break;
            case '4':
                sscanf(optarg, "%d", &M);
                break;
            case '5':
                sscanf(optarg, "%d", &efConstruction);
                break;
            case '6':
                sscanf(optarg, "%d", &efSearch);
                break;
            case '7':
                index_filepath = optarg;
                break;
            case '8':
                save_index = true;
                break;
            case '9':
                if (strcmp(optarg, "early-stop-training") == 0) {
                    mode = OBSERVATION_DATA_GENERATION;
                } else if (strcmp(optarg, "early-stop-testing") == 0) {
                    mode = DARTH_TESTING;
                } else if (strcmp(optarg, "no-early-stop") == 0) {
                    mode = NO_EARLY_TERMINATION_TESTING;
                } else if (strcmp(optarg, "naive-early-stop-testing") == 0) {
                    mode = BASELINE_TESTING;
                } else if (strcmp(optarg, "optimal-early-stop-testing") == 0) {
                    mode = OPTIMAL_RESULTS_GENERATION;
                } else if (strcmp(optarg, "laet-early-stop-testing") == 0) {
                    mode = LAET_TESTING;
                } else if (strcmp(optarg, "raet-early-stop-testing") == 0) {
                    mode = RAET_TESTING;
                }else if (strcmp(optarg, "raet-get-stability-times-testing") == 0) {
                    mode = RAET_GET_STABILITY_TIMES;
                }else if (strcmp(optarg, "raet-CNS-fea-training") == 0) {
                    mode = RAET_CNS_DATA_GENERATION;
                }else if (strcmp(optarg, "optimal-efSearch-per-query") == 0) {
                    mode = OPTIMAL_EFSEARCH_PER_QUERY;
                }else if (strcmp(optarg, "raet-CNS-early-stop-testing") == 0) {
                    mode = RAET_CNS_TESTING;
                }else if (strcmp(optarg, "raet-select-model") == 0) {
                    mode = RAET_SELECT_MODEL;
                }else if (strcmp(optarg, "raet-select-model-data-gen") == 0) {
                    mode = RAET_SELECT_MODEL_DATA_GEN;
                }else if (strcmp(optarg, "raet-query-select-model") == 0) {
                    mode = RAET_QUERY_SELECT_MODEL;
                }
                else {
                    printf("Unknown running mode: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'a':
                sscanf(optarg, "%f", &target_recall);
                break;
            case 'b':
                sscanf(optarg, "%d", &initial_prediction_interval);
                break;
            case 'c':
                predictor_model_path = optarg;
                break;
            case 'd':
                sscanf(optarg, "%d", &logging_interval);
                break;
            case 'e':
                sscanf(optarg, "%d", &fixed_amount_of_search);
                break;
            case 'f':
                sscanf(optarg, "%f", &prediction_multiplier);
                break;
            case 'g':
                sscanf(optarg, "%d", &min_prediction_interval);
                break;
            case 'h':
                per_prediction_logging = true;
                break;
            case 'i':
                verbose = true;
                break;
            case 'j':
                if (strcmp(optarg, "training") == 0) {
                    query_type = TRAINING;
                } else if (strcmp(optarg, "validation") == 0) {
                    query_type = VALIDATION;
                } else if (strcmp(optarg, "testing") == 0) {
                    query_type = TESTING;
                } else if (strcmp(optarg, "noisy-testing") == 0) {
                    query_type = NOISY_TESTING;
                } else {
                    printf("Unknown query type: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'k':
                dataset_dir_prefix = optarg;
                break;
            case 'l':
                // sscanf(optarg, "%f", &gnoise);
                noise_perc = optarg;
                break;
            case 'm':
                printf("This is the driver testing code for the DARTH paper.\n");
                exit(EXIT_SUCCESS);
            case 'n':
                sscanf(optarg, "%d", &stability_times);
                break;
            case 'o':
                sscanf(optarg, "%d", &train_CNS);
                break;
            case 'p':
                stability_log_path = optarg;
                break;
            case 'q':
                sscanf(optarg, "%d", &stability_times_r1);
                break;
            case 'r':
                predictor_model_path_select_CNS = optarg;
                break;
            case 's':
                model_selector_path = optarg;
                break;
            default:
                printf("Unknown option: %c\n", c);
                exit(EXIT_FAILURE);
        }
    }

    if (initial_prediction_interval < min_prediction_interval) {
        printf("Initial prediction interval should be greater than or equal to min prediction interval\n");
        exit(EXIT_FAILURE);
    }

    // print a log with the parameters
    printf(">>Parameters:\n");
    printf("   dataset: %s\n", dataset);
    printf("   nQ: %ld\n", nQ);
    printf("   output: %s\n", output);
    printf("   M: %d, efConstruction: %d, efSearch: %d, k: %ld\n", M, efConstruction, efSearch, k);
    printf("   index_filepath: %s\n", index_filepath);
    printf("   save_index: %s\n", save_index == true ? "yes" : "no");
    printf("   mode: %s\n", running_mode_str[mode]);
    printf("   target_recall: %f\n", target_recall);
    printf("   logging_interval: %d\n", logging_interval);
    printf("   initial_prediction_interval: %d, min_prediction_interval: %d\n", initial_prediction_interval,
           min_prediction_interval);
    printf("   predictor_model_path: %s\n", predictor_model_path);
    printf("   per_prediction_logging: %s\n", per_prediction_logging == true ? "yes" : "no");
    printf("   query_type: %s\n", query_type_str[query_type]);
    printf("   dataset_dir_prefix: %s\n", dataset_dir_prefix);
    printf("   verbose: %s\n", verbose == true ? "yes" : "no");
    printf("   gnoise perc: %s\n", noise_perc);
    printf("   [LAET] fixed_amount_of_search: %d, prediction_multiplier: %f\n", fixed_amount_of_search,
           prediction_multiplier);
    printf("   stability_times: %d\n", stability_times);
    printf("   stability_times_r1: %d\n", stability_times_r1);
    printf("   找每个查询点最小CNS时才用到，其他情况默认值为500无用。train_CNS: %d\n", train_CNS);
    printf("   stability_log_path: %s\n", stability_log_path.c_str());

    VectorDataLoader vector_dataloader(dataset, query_type, noise_perc, dataset_dir_prefix);
    vector_dataloader.initializeDataMaps();

    // Load database vectors
    size_t d, n;
    float *vecsDB = vector_dataloader.loadDB(&d, &n);
    printf(">>%s DB loaded: d = %ld, n = %ld. Elapsed = %.3fs\n", dataset, d, n, (elapsed() - t0));

    // Load query vectors
    size_t dQ, nQ_all;
    float *vecsQ_all = vector_dataloader.loadQueries(&dQ, &nQ_all);
    assert(dQ == d);

    // Load ground-truth results
    size_t nQ2, k_all;
    int *gt_int_all = vector_dataloader.loadQueriesGroundtruths(&k_all, &nQ2);
    assert(nQ_all == nQ2);

    // Load ground-truth distances，只在测试集使用groundtruth_distance，为不影响后续代码，其他查询类型暂时设置为0
    size_t nQ3, k_all2;
    float *gt_dist_all = nullptr;   // 先声明
//    if(query_type == TESTING){
//        gt_dist_all =vector_dataloader.loadQueriesGroundtruthDistances(&k_all2, &nQ3);
//        assert(nQ_all == nQ3);
//    }else{
//        gt_dist_all = new float[k_all * nQ_all];
//        memset(gt_dist_all, 0, sizeof(float) * k_all * nQ_all);
//    }

    gt_dist_all = new float[k_all * nQ_all];
    memset(gt_dist_all, 0, sizeof(float) * k_all * nQ_all);

//它将 ground truth（GT）矩阵从 int 类型转换为 FAISS 专用的 idx_t 类型。
    faiss::idx_t *gt_all = new faiss::idx_t[k_all * nQ_all];
    for (faiss::idx_t i = 0; i < k_all * nQ_all; i++) {
        gt_all[i] = static_cast<faiss::idx_t>(gt_int_all[i]);
    }
    delete[] gt_int_all;

    float *vecsQ;
    faiss::idx_t *gt_k_all, *indicesQ;
    float *gt_k_dist_all;


    get_queries(vecsQ_all, nQ_all, d, gt_all, gt_dist_all, k_all, nQ, &vecsQ, &indicesQ, &gt_k_all, &gt_k_dist_all,
                false);

    printf(">>k=%ld, k_all=%ld\n", k, k_all);
    assert(k_all >= k);
    faiss::idx_t *gt = new faiss::idx_t[k * nQ];
    float *gt_dist = new float[k * nQ];
    for (faiss::idx_t i = 0; i < nQ; i++) {
        for (faiss::idx_t j = 0; j < k; j++) {
            gt[i * k + j] = gt_k_all[i * k_all + j];
            gt_dist[i * k + j] = gt_k_dist_all[i * k_all + j];
        }
    }

    printf(">>Query Vectors and GT loaded: d = %ld, nQ = %ld, k = %ld, Elapsed = %.3fs\n", dQ, nQ, k, (elapsed() - t0));

    double index_build_start = elapsed();
    faiss::IndexHNSWFlat index(d, M);

    if (save_index) {
        if (index_filepath == NULL) {
            printf(">>Index file path is required to save the index\n");
            exit(EXIT_FAILURE);
        }

        index.hnsw.efConstruction = efConstruction;
        index.hnsw.efSearch = efSearch;
        index.add(n, vecsDB);
        faiss::write_index(&index, index_filepath);

        printf(">>Index saved to %s\n", index_filepath);
    } else {
        if (index_filepath) {
            index = *dynamic_cast<faiss::IndexHNSWFlat *>(faiss::read_index(index_filepath));
            index.hnsw.efSearch = efSearch;
            printf(">>Index loaded from %s\n", index_filepath);
        } else {
            index.hnsw.efConstruction = efConstruction;
            index.hnsw.efSearch = efSearch;
            index.add(n, vecsDB);
        }
    }

    double index_build_time = elapsed() - index_build_start;
    printf(">>Index build time: %.3fs\n", index_build_time);

    //每个数据点的入度=出度=度数,没有用处，每个数据点的度数都相同，sift为64
    std::vector<int> out_degree_L0(n, 0);
    auto& hnsw = index.hnsw;
    for (faiss::idx_t i = 0; i < n; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);  // 只看底层
        out_degree_L0[i] = static_cast<int>(end - begin);
    }
    // 每个数据点的跳数特征===== hop data (allocated once, outside search) =====
    std::vector<int> node_hop(n, 0);          // hop 值（不用初始化）
    std::vector<int> node_hop_qid(n, -1);     // 时间戳，必须初始化一次
    printf(">>Out-degree (layer 0) precomputed.\n");

    //预归一化
    // ===== ⭐ 1. 为 DB & Query 各创建一个归一化副本 =====
    float* norm_db = new float[n * d];
    float* norm_queries = new float[nQ * d];
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
    printf(">>预归一化完成\n");



    faiss::idx_t *I = new faiss::idx_t[nQ * k];
    float *D = new float[nQ * k];

    // Perform the search
    double search_start_time = elapsed();
    //将度数特征加入到data_manager中
    faiss::DeclarativeRecallDataManager data_manager(D, I, gt, gt_dist, nQ, d, k, vecsQ, output, vecsDB, n, gt_k_all,
                                                     k_all, &out_degree_L0, &node_hop, &node_hop_qid,
                                                     norm_queries,
                                                     norm_db);
    switch (mode) {
        case OBSERVATION_DATA_GENERATION: {
            faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager,logging_interval);
            search_monitor.init_log_file();
            index.search_declarative_recall_data_generation(nQ, vecsQ, k, D, I, search_monitor);
            search_monitor.close_log_file();
            break;
        }
        case DARTH_TESTING: {
            faiss::DARTHPredictorHNSW recall_predictor(data_manager, target_recall, initial_prediction_interval,
                                                       min_prediction_interval, per_prediction_logging,
                                                       predictor_model_path);
            recall_predictor.init_log_file();
            index.search_DARTH(nQ, vecsQ, k, D, I, recall_predictor);
            recall_predictor.close_log_file();
            break;
        }
        case NO_EARLY_TERMINATION_TESTING: {
            faiss::DeclarativeRecallDataCollectorHNSW searchMonitor(data_manager, INTERVAL_DISABLED_VALUE);
            searchMonitor.init_log_file();
            index.search_baseline(nQ, vecsQ, k, D, I, searchMonitor);
            searchMonitor.close_log_file();
            break;
        }
        case BASELINE_TESTING: {
            int distances_to_stop_at = initial_prediction_interval;
            faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, INTERVAL_DISABLED_VALUE,
                                                                     distances_to_stop_at);
            search_monitor.init_log_file();
            index.search_baseline(nQ, vecsQ, k, D, I, search_monitor);
            search_monitor.close_log_file();
            break;
        }
        case OPTIMAL_RESULTS_GENERATION: {
            faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, INTERVAL_WHEN_RECALL_UPDATES_VALUE);
            search_monitor.init_log_file();
            index.search_baseline(nQ, vecsQ, k, D, I, search_monitor);
            search_monitor.close_log_file();
            break;
        }
        case LAET_TESTING: {
            faiss::LAETPredictorHNSW laet_predictor(data_manager, fixed_amount_of_search, prediction_multiplier,
                                                    predictor_model_path);
            laet_predictor.init_log_file();
            index.search_LAET(nQ, vecsQ, k, D, I, laet_predictor);
            laet_predictor.close_log_file();
            break;
        }
        //预测准确率方法,search_from_candidates_RAET_base
        case RAET_TESTING: {
            faiss::RAETPredictorHNSW raet_predictor(data_manager, target_recall, stability_times,stability_times_r1,per_prediction_logging,
                                                       predictor_model_path);
            raet_predictor.init_log_file();
            index.search_RAET(nQ, vecsQ, k, D, I, raet_predictor);
            raet_predictor.close_log_file();
            break;
        }
        //获取稳定次数，将稳定次数输出到文件。search_from_candidates_RAET_GET_STABILITY_TIMES
        case RAET_GET_STABILITY_TIMES: {
            faiss::RAETPredictorHNSW raet_predictor(data_manager, target_recall, stability_times,stability_times_r1,per_prediction_logging,
                                                    predictor_model_path);
            raet_predictor.init_log_file();
//            std::string stability_log_path = "/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main-data/统计文件/CNS137-stability_counts0.98.csv";
//            std::string stability_log_path = "";
            index.search_RAET_GET_STABILITY_TIMES(nQ, vecsQ, k, D, I, raet_predictor,stability_log_path);
            raet_predictor.close_log_file();
            break;
        }
        //预测CNS方法  search_from_candidates_RAET_CNS_test
        case RAET_CNS_TESTING: {
            faiss::RAETCNSPredictorHNSW raet_CNS_predictor(data_manager, target_recall,per_prediction_logging,
                                                    predictor_model_path);
            raet_CNS_predictor.init_log_file();
            index.search_RAET_CNS(nQ, vecsQ, k, D, I, raet_CNS_predictor);
            raet_CNS_predictor.close_log_file();
            break;
        }
        //生成预测CNS的特征数据  search_from_candidates_RAET_CNS_data_generation
        case RAET_CNS_DATA_GENERATION: {
            faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, logging_interval);
            search_monitor.init_CNS_log_file();
            index.search_RAET_CNS_data_generation(nQ, vecsQ, k, D, I, search_monitor);
            search_monitor.close_log_file();
            break;
        }
        //根据搜索过程中的状态变化，进行模型的选择，是使用预测准确率方法还是预测CNS方法
        case RAET_SELECT_MODEL: {
            faiss::RAETModelSelectPredictorHNSW raet_model_select_predictor(data_manager, target_recall, stability_times,stability_times_r1,per_prediction_logging,
                                                                            predictor_model_path,predictor_model_path_select_CNS,model_selector_path);
            raet_model_select_predictor.init_log_file();
            index.search_RAET_Select(nQ, vecsQ, k, D, I, raet_model_select_predictor);
            raet_model_select_predictor.close_log_file();
            break;
        }
        //模型选择方法的特征收集器
        case RAET_SELECT_MODEL_DATA_GEN: {
            faiss::RAETModelSelectPredictorHNSW raet_model_select_data_generator(data_manager, target_recall, stability_times,stability_times_r1,per_prediction_logging,
                                                                            predictor_model_path,predictor_model_path_select_CNS,model_selector_path);
            raet_model_select_data_generator.init_Select_Model_log_file();
            index.search_RAET_Select_data_generation(nQ, vecsQ, k, D, I, raet_model_select_data_generator);
            raet_model_select_data_generator.close_log_file();
            break;
        }
        case RAET_QUERY_SELECT_MODEL: {
            faiss::RAETPredictorHNSW raet_predictor(data_manager, target_recall, stability_times,stability_times_r1,per_prediction_logging,predictor_model_path);
            faiss::RAETCNSPredictorHNSW raet_CNS_predictor(data_manager, target_recall,per_prediction_logging,predictor_model_path_select_CNS);
            faiss::RAETModelQuerySelectPredictorHNSW query_model_selector(data_manager,model_selector_path);
            raet_predictor.init_log_file();
            index.search_RAET_Query_Select(nQ, vecsQ, k, D, I, raet_predictor,raet_CNS_predictor,query_model_selector);
            raet_predictor.close_log_file();
            break;
        }
        //获取每个查询点达到要求准确率的最小CNS，并保存到文件
        case OPTIMAL_EFSEARCH_PER_QUERY: {
            int ef_min=k;
            int ef_max=train_CNS;
            std::vector<int> min_ef_per_query(nQ, -1);
            // 二分搜索的 helper
            auto run_search_for_ef = [&](faiss::idx_t q, int ef) -> float {
                index.hnsw.efSearch = ef;

                // 创建只搜索单 query 的 search monitor
                faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, INTERVAL_DISABLED_VALUE);

                // 搜索
                index.search_baseline(1, vecsQ + q * d, k,
                                      D + q * k, I + q * k, search_monitor);

                // 返回 recall
                return data_manager.get_recallk(q);
            };

            for (faiss::idx_t q = 0; q < nQ; q++) {

                // ---------- ① 先用 train_CNS 跑一遍 ----------
                float recall_with_train = run_search_for_ef(q, train_CNS);

                if (recall_with_train < target_recall) {

                    // 兜底：train_CNS 也达不到，就直接写 train_CNS
                    min_ef_per_query[q] = train_CNS;

                } else {

                    // ---------- ② 否则正常做二分 ----------
                    int l = ef_min, r = train_CNS;
                    int ans = train_CNS;   // 注意：初始化成 train_CNS

                    while (l <= r) {
                        int mid = l + (r - l) / 2;
                        float recall_q = run_search_for_ef(q, mid);

                        if (recall_q >= target_recall) {
                            ans = mid;
                            r = mid - 1;
                        } else {
                            l = mid + 1;
                        }
                    }

                    min_ef_per_query[q] = ans;
                }
            }

            // 可选：保存 CSV
            FILE *fout = fopen(output, "w");
//            FILE *fout = fopen("/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main-data/data/et_training_data/test_logging/SIFT1M/k100/min_ef_per_query_target_reacall.csv", "w");
            fprintf(fout, "qid,min_efSearch\n");
            //向文本中写入数据
            for (faiss::idx_t q = 0; q < nQ; q++) {
                fprintf(fout, "%ld,%d\n", q, min_ef_per_query[q]);
            }
            fclose(fout);

            break;
        }

        default:
            printf(">>Unknown running mode: %d\n", mode);
            exit(EXIT_FAILURE);
    }

    double search_time = elapsed() - search_start_time;

    printf(
            "\n\nIndex[M=%d, efC=%d, efS=%d]IndexTime: %lfs, SearchTime: %lfs, "
            "TotalTime: %lfs, Recall@%ld: %.4f\n",
            M, index.hnsw.efConstruction, index.hnsw.efSearch, index_build_time,
            search_time, (elapsed() - t0), k,
            recall_at_k_avg(gt, I, D, k, nQ, gt_k_all, k_all, verbose));

    delete[] gt_k_all;
    delete[] gt_dist_all;
    delete[] gt_all;
    delete[] vecsDB;
    delete[] vecsQ;
    delete[] norm_db;
    delete[] norm_queries;
    delete[] gt;
    delete[] I;
    delete[] D;

    return 0;
}
