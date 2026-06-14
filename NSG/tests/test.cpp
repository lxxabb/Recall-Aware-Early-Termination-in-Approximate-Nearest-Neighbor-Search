//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[(size_t) num * (size_t) dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}
float cal_recall_one_point(std::vector<unsigned> results, std::vector<unsigned> true_data, unsigned k)
{
    float acc = 0;
    for (size_t j = 0; j < k; j++)
    {
        for (size_t m = 0; m < k; m++)
        {
            if (results[j] == true_data[m])
            {
                acc++;
                break;
            }
        }
    }

    return acc / k;
}
void save_result(const char *filename,
                 std::vector<std::vector<unsigned> > &results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned) results[i].size();
        out.write((char *) &GK, sizeof(unsigned));
        out.write((char *) results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

//读取ivces文件，就是ground_truth文件
void load_ivecs_data(const char *filename,
                     std::vector<std::vector<unsigned>> &results, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    //std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    results.resize(num);
    for (unsigned i = 0; i < num; i++) results[i].resize(dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) results[i].data(), dim * 4);
    }
    in.close();
}

float cal_recall(std::vector<std::vector<unsigned>> results, std::vector<std::vector<unsigned>> true_data, unsigned num,
                 unsigned k) {
    //统计进行模型预测时，模型调用情况
    //    std::ofstream csvFile("/home/extra_home/lxx23125236/ann-data/PGAS/query_NSSG_k_100_预测0.99准确率停止时各个数据点准确率情况.csv", std::ios::app);

    float mean_acc = 0;
    for (size_t i = 0; i < num; i++) {
        float acc = 0;
        for (size_t j = 0; j < k; j++) {
            for (size_t m = 0; m < k; m++) {
                if (results[i][j] == true_data[i][m]) {
                    acc++;
                    break;
                }
            }
        }
        //        csvFile << acc / k;
        //        csvFile << std::endl;
        mean_acc += acc / k;
    }
    return mean_acc / num;
}

// 一次性把单列 CSV 读成 vector<double>
inline std::vector<double> load_recall_list(const std::string &csv) {
    std::ifstream fin(csv);
    if (!fin) throw std::runtime_error("cannot open every_point_recall csv: " + csv);
    std::vector<double> v;
    double x;
    while (fin >> x) v.push_back(x);
    return v;
}
//输入参数：0cpp文件 1base数据集，2查询数据集，3图索引，4L,5K,6Groundtruth路径,7 用户要求准确率(训练模式随便填)
//8模式（训练集收集特征"train_data"，测试集搜索"test"） 9训练集输出文件路径或者测试集模型文件路径 10稳定性参数 11 k=10时需要的user_require_recall
//构建训练数据：
//搜索
struct Args {
    std::string base_data;
    std::string query_data;
    std::string graph_path;
    std::string groundtruth_path;

    unsigned L = 0;
    unsigned K = 0;

    float required_recall = 0.0f;

    std::string mode;          // train_data / test / get_train_CNS / ...
    std::string io_path;       // csv 或 model path

    unsigned stability_times = 0;
    std::string every_point_recall_path; // 仅 K=10 时使用
};
Args parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> kv;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) != 0)
            throw std::runtime_error("Invalid arg: " + arg);

        auto pos = arg.find('=');
        if (pos == std::string::npos)
            throw std::runtime_error("Arg must be --key=value: " + arg);

        kv[arg.substr(2, pos - 2)] = arg.substr(pos + 1);
    }

    Args a;
    a.base_data  = kv.at("base");
    a.query_data = kv.at("query");
    a.graph_path = kv.at("graph");
    a.groundtruth_path = kv.at("gt");

    a.L = std::stoul(kv.at("L"));
    a.K = std::stoul(kv.at("K"));

    a.required_recall = std::stof(kv.at("recall"));
    a.mode = kv.at("mode");
    a.io_path = kv.at("io");

    if (kv.count("stability"))
        a.stability_times = std::stoul(kv.at("stability"));

    if (kv.count("per_query_recall"))
        a.every_point_recall_path = kv.at("per_query_recall");

    return a;
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    // === 数据加载 ===
    float* data_load = nullptr;
    unsigned points_num, dim;
    load_data(args.base_data.c_str(), data_load, points_num, dim);

    float* query_load = nullptr;
    unsigned query_num, query_dim;
    load_data(args.query_data.c_str(), query_load, query_num, query_dim);
    assert(dim == query_dim);

    // === Groundtruth ===
    std::vector<std::vector<unsigned>> true_load;
    unsigned true_num, true_dim;
    load_ivecs_data(args.groundtruth_path.c_str(), true_load, true_num, true_dim);

    /* ========= 索引初始化 ========= */
    efanna2e::IndexNSG index(dim, points_num,
                             efanna2e::FAST_L2, nullptr);
    index.Load(args.graph_path.c_str());
    index.OptimizeGraph(data_load);

    index.InitTimeSpendModel();
    index.InitDistCount();
    index.InitModelPreCount();

    index.load_ivecs_data(argv[6], true_num, true_dim);

    /* ========= 搜索参数 ========= */
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", args.L);
    paras.Set<unsigned>("P_search", args.L);

    /* ========= 模式分发 ========= */
    if (args.mode == "train_data") {
        run_train_data(index, args,
                       query_load, query_num, dim,
                       paras);
    }
    else if (args.mode == "test") {
        run_test(index, args,
                 query_load, query_num, dim,
                 paras, true_load);
    }
    else if (args.mode == "get_train_CNS") {
        run_get_train_CNS(index, args,
                          query_load, query_num, dim,
                          paras, true_load);
    }
    else if (args.mode == "find_every_point_min_L") {
        run_find_every_point_min_L(index, args,
                                   query_load, query_num, dim,
                                   paras, true_load);
    }
    else if (args.mode == "get_StablizeCount") {
        run_get_stablize_count(index, args,
                               query_load, query_num, dim,
                               paras);
    }
    else {
        throw std::runtime_error("Unknown mode: " + args.mode);
    }

    if (argc < 10) {
        std::cout << argv[0]
                  << "./run data_file query_file ssg_path L K groundtruth_path user_require_recall train_data/test csvfile_path/modelfile_path stability_times every_point_reacll"
                  << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);
    float *query_load = NULL;
    unsigned query_num, query_dim;
    load_data(argv[2], query_load, query_num, query_dim);
    assert(dim == query_dim);

    std::vector<std::vector<unsigned>> true_load;
    unsigned true_dim, true_num;
    //groundtruth
    load_ivecs_data(argv[6], true_load, true_num, true_dim);

    unsigned L = (unsigned) atoi(argv[4]);
    unsigned K = (unsigned) atoi(argv[5]);

    if (L < K) {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }

    // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
    // align the data before build query_load = efanna2e::data_align(query_load,
    // query_num, query_dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
    index.Load(argv[3]);
    index.OptimizeGraph(data_load);
    index.load_ivecs_data(argv[6], true_num, true_dim);
    //初始化模型预测时间
    index.InitTimeSpendModel();
    //初始化距离计算次数
    index.InitDistCount();
    //初始化模型预测次数
    index.InitModelPreCount();

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    std::vector<std::vector<unsigned> > res(query_num);
    for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

    //要求准确率设置
    float required_accuracy = std::stof(argv[7]);
    std::cout << "要求准确率为： " << required_accuracy << std::endl;

    unsigned stability_times = (unsigned) atoi(argv[10]);

    std::string L_type(argv[8]);


    if (L_type == "train_data") {
        for (unsigned i = 0; i < query_num; i++) {
            index.SearchWithOptGraph_train_data(query_load + i * dim, K, paras, res[i].data(), required_accuracy, i,
                                                argv[9]);
        }
        std::cout << "特征和标签已经收集完毕，生成的CSV文件存储位置为： " << argv[9] << std::endl;
    }
    //获取达到要求准确率的最小CNS
    if (L_type == "get_train_CNS") {
        std::vector<std::vector<unsigned>> res(query_num);
        for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

        std::set<unsigned> visited;
        std::map<unsigned, float> L_acc; // 记录已测试L及其准确率
        float acc_set = std::stof(argv[5]);
        //                unsigned K = /* 你的K值 */; // 确保K已正确定义
        unsigned L_min = 0x7fffffff;
        // 阶段一：指数扩展找到初始上界
        unsigned low = K;
        unsigned high = K;
        bool found = false;
        while (true) {
            if (visited.count(high)) {
                if (L_acc[high] >= acc_set) {
                    found = true;
                    break;
                } else {
                    low = high + 1;
                    high *= 2;
                }
            } else {
                for (unsigned i = 0; i < query_num; i++) {
                    index.SearchWithOptGraph_get_everyPoint_CNS(query_load + i * dim, K, paras, res[i].data(), high);
                }
                float acc = cal_recall(res, true_data, query_num, K);
                std::cout << "CNS=" << high << "时" << K << " NN准确率: " << acc << std::endl;

                visited.insert(high);
                L_acc[high] = acc;
                if (acc >= acc_set) {
                    found = true;
                    break;
                } else {
                    low = high + 1;
                    high *= 2;
                }
            }
        }
        // 阶段二：二分查找精确确定L_min
        unsigned left = K;
        unsigned right = right = high;
        L_min = high;
        while (left <= right) {
            unsigned mid = left + (right - left) / 2;
            float acc;
            if (visited.count(mid)) {
                acc = L_acc[mid];
            } else {
                for (unsigned i = 0; i < query_num; i++) {
                    index.SearchWithOptGraph_get_everyPoint_CNS(query_load + i * dim, K, paras, res[i].data(), mid);
                }
                acc = cal_recall(res, true_data, query_num, K);
                std::cout << "CNS=" << mid << "时" << K << " NN准确率: " << acc << std::endl;
                visited.insert(mid);
                L_acc[mid] = acc;
            }
            if (acc >= acc_set) {
                if (mid < L_min) {
                    L_min = mid;
                }
                right = mid - 1; // 尝试更小的L
            } else {
                left = mid + 1; // 需要更大的L
            }
        }
        std::cout << "当CNS=" << L_min << "时，在训练数据集上达到平均要求准确率" << std::endl;
    } else if (L_type == "find_every_point_min_L") {
        //设置准确率，找出每个数据点最小候选邻居集合
        // sift在循环外部创建并初始化输出文件 k=10
        std::ofstream output_file(argv[6], std::ios::app);
        // std::ofstream output_file("/home/extra_home/lxx23125236/model_predict_CNS/query_TOGG_L_min_to_99acc.csv");
        // sift k=100
        //            std::ofstream output_file("/home/extra_home/lxx23125236/model_predict_CNS/label/sift/k_100_acc_99/query_TOGG_L_min_to_99acc.csv");
        if (!output_file.is_open()) {
            std::cerr << "Error: Failed to create output file!" << std::endl;
            // 这里可以根据实际情况处理错误（如退出程序）
        }
        unsigned avg_CNS = (unsigned) atoi(argv[7]);

        index.InitDistCount();
        std::vector<std::vector<unsigned>> res(query_num);
        for (unsigned i = 0; i < query_num; i++) res[i].resize(K);
        for (unsigned i = 0; i < query_num; i++) {
            std::set<unsigned> visited;
            std::map<unsigned, float> L_acc; // 记录已测试L及其准确率
            float acc_set = std::stof(argv[5]);
            //                unsigned K = /* 你的K值 */; // 确保K已正确定义
            unsigned L_min = 0x7fffffff;
            // 阶段一：指数扩展找到初始上界
            unsigned low = K;
            unsigned high = K;
            bool found = false;
            while (true) {
                if (visited.count(high)) {
                    if (L_acc[high] >= acc_set) {
                        high = std::min(high, avg_CNS);
                        found = true;
                        break;
                    } else {
                        low = high + 1;
                        high *= 2;
                    }
                } else {
                    index.SearchWithOptGraph_get_everyPoint_CNS(query_load + i * dim, K, paras, res[i].data(), high);
                    float acc = cal_recall_one_point(res[i], true_data[i], K);
                    visited.insert(high);
                    L_acc[high] = acc;
                    //                        std::cout << i << " Testing L=" << high << " Acc: " << acc << std::endl;
                    if (acc >= acc_set) {
                        high = std::min(high, avg_CNS);
                        found = true;
                        break;
                    } else {
                        low = high + 1;
                        high *= 2;
                    }
                }
                // 防止无界增长，设置最大阈值  sift 设置120  k=100 设置300  gist设置500(k=10)   gist设置1000(k=100)  glove设置25000
                if (high > 2 * avg_CNS) {
                    // 根据实际情况调整
                    break;
                }
            }
            if (!found) {
                //output_file << i << "," << 100 << "\n";  // CSV格式
                //sift k=10
                output_file << avg_CNS << std::endl;
                //sift k=100
                //output_file << 144 << "\n";
                // output_file << 5000 << "\n";
                continue;
            }
            // 阶段二：二分查找精确确定L_min
            unsigned left = K;
            unsigned right = right = std::min(high, avg_CNS);
            L_min = high;

            while (left <= right) {
                unsigned mid = left + (right - left) / 2;
                float acc;
                if (visited.count(mid)) {
                    acc = L_acc[mid];
                } else {
                    index.SearchWithOptGraph_get_everyPoint_CNS(query_load + i * dim, K, paras, res[i].data(), mid);
                    acc = cal_recall_one_point(res[i], true_data[i], K);
                    visited.insert(mid);
                    L_acc[mid] = acc;
                    //                        std::cout << i << " Testing L=" << mid << " Acc: " << acc << std::endl;
                }

                if (acc >= acc_set) {
                    if (mid < L_min) {
                        L_min = mid;
                    }
                    right = mid - 1; // 尝试更小的L
                } else {
                    left = mid + 1; // 需要更大的L
                }
            }
            //                output_file << i << "," << L_min << "\n";  // CSV格式
            output_file << L_min << std::endl;
            // 输出结果
            //                std::cout << "Query " << i << " L_min: " << L_min << std::endl;
        }
        std::cout << "找到每个数据点最小CNS，CSV文件保存路径为" << argv[6] << std::endl;
    }
        //测试数据的模型预测
    else if (L_type == "test") {
        //回归模型位置
        std::string single_model_regression_path(argv[9]);
        // const std::string single_model_regression_path = "/home/extra_home/lxx23125236/ann-data/PGAS/sift_train/PGAS_LGB_sift_new_1w_1.txt";
        //回归模型加载
        index.Predict_recall_LoadModel_Regression(single_model_regression_path);
        if (K == 100) {
            auto s = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < query_num; i++) {
                // index.SearchWithOptGraph_pre_recall(query_load + i * dim, K, paras, res[i].data(),required_accuracy);
                //郝老师方法，更少的模型预测次数
                index.SearchWithOptGraph_pre_recall_lessPre(query_load + i * dim, K, paras, res[i].data(),
                                                            required_accuracy, stability_times);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cerr << "总搜索时间: " << diff.count() << std::endl;
        } else if (K == 10) {
            /*===== 新增 =====*/
            std::vector<double> acc_table;          // 默认为空
            // 只在 K=10 时加载
            acc_table = load_recall_list(argv[11]);
            if (acc_table.size() != query_num)
                throw std::runtime_error("CSV row num != query_num");

            auto s = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < query_num; i++) {
                float req_acc = static_cast<float>(acc_table[i]); // 逐 Query 取值
                // index.SearchWithOptGraph_pre_recall(query_load + i * dim, K, paras, res[i].data(),required_accuracy);
                //郝老师方法，更少的模型预测次数
                index.SearchWithOptGraph_pre_recall_lessPre(query_load + i * dim, K, paras, res[i].data(), req_acc,
                                                            stability_times);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cerr << "总搜索时间: " << diff.count() << std::endl;
        }

        //计算模型预测操作所花费时间
        int64_t TimeSpendModel = index.GetTimeSpendModel();
        // 将纳秒转换为秒
        double TimeSpendModelInSeconds = static_cast<double>(TimeSpendModel) / 1e9;
        // 输出模型预测花费的时间（秒）
        std::cout << "模型预测花费时间： " << TimeSpendModelInSeconds << " 秒" << std::endl;

        size_t ModelPreCount = index.GetModelPreCount();
        std::cout << "模型预测次数: " << ModelPreCount << std::endl;

        size_t DistCount = index.GetDistCount();
        std::cout << "距离计算次数: " << DistCount << std::endl;

        float recall = cal_recall(res, true_load, query_num, K);
        std::cout << K << " NN 准确率: " << recall << std::endl;
    } else if (L_type == "get_StablizeCount") {
        for (unsigned i = 0; i < query_num; i++) {
            index.SearchWithOptGraph_stablize(query_load + i * dim, K, paras, res[i].data(), required_accuracy, i,
                                              argv[9]);
        }
        std::cout << "稳定次数收集完毕，生成的CSV文件存储位置为： " << argv[9] << std::endl;
    }
    return 0;
}
void run_train_data(
        efanna2e::IndexNSG& index,
        const Args& args,
        float* query,
        unsigned query_num,
        unsigned dim,
        efanna2e::Parameters& paras
) {
    std::vector<std::vector<unsigned>> res(
            query_num, std::vector<unsigned>(args.K)
    );

    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_train_data(
                query + i * dim,
                args.K,
                paras,
                res[i].data(),
                args.required_recall,
                i,
                args.io_path.c_str()
        );
    }

    std::cout << "特征与标签已收集，CSV 路径: "
              << args.io_path << std::endl;
}
