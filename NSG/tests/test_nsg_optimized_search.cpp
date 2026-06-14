//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>

void load_data(const char *filename, float *&data, unsigned &num,
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
void load_train_or_test_data(const char *filename,float *&data,unsigned &num,unsigned &dim,unsigned query_num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }
    // 1. 读取向量维度
    in.read((char*)&dim, 4);
    // 2. 计算文件中真实向量数
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();
    unsigned total_num = (unsigned)(fsize / ((dim + 1) * 4));
    // 3. 实际读取数量
    num = std::min(query_num, total_num);
    // 4. 分配内存
    data = new float[(size_t)num * dim];
    // 5. 回到文件开头
    in.seekg(0, std::ios::beg);
    // 6. 只读取 num 条
    for (unsigned i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);                 // 跳过 dim
        in.read((char*)(data + i * dim), dim * 4); // 读向量
    }
    in.close();
}

float cal_recall_one_point(std::vector<unsigned> results, std::vector<unsigned> true_data, unsigned k) {
    float acc = 0;
    for (size_t j = 0; j < k; j++) {
        for (size_t m = 0; m < k; m++) {
            if (results[j] == true_data[m]) {
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
//void load_ivecs_data(const char *filename,
//                     std::vector<std::vector<unsigned>> &results, unsigned &num, unsigned &dim) {
//    std::ifstream in(filename, std::ios::binary);
//    if (!in.is_open()) {
//        std::cout << "open file error" << std::endl;
//        exit(-1);
//    }
//    in.read((char *) &dim, 4);
//    //std::cout<<"data dimension: "<<dim<<std::endl;
//    in.seekg(0, std::ios::end);
//    std::ios::pos_type ss = in.tellg();
//    size_t fsize = (size_t) ss;
//    num = (unsigned) (fsize / (dim + 1) / 4);
//    results.resize(num);
//    for (unsigned i = 0; i < num; i++) results[i].resize(dim);
//
//    in.seekg(0, std::ios::beg);
//    for (size_t i = 0; i < num; i++) {
//        in.seekg(4, std::ios::cur);
//        in.read((char *) results[i].data(), dim * 4);
//    }
//    in.close();
//}
void load_ivecs_data(const char *filename,std::vector<std::vector<unsigned>> &results,unsigned &num,unsigned &dim,unsigned query_num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }
    // 1. 读取向量维度
    in.read((char*)&dim, 4);
    // 2. 计算文件中真实向量数
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();
    unsigned total_num = (unsigned)(fsize / ((dim + 1) * 4));
    // 3. 实际读取数量
    num = std::min(query_num, total_num);
    // 4. 分配结果空间
    results.clear();
    results.resize(num);
    for (unsigned i = 0; i < num; i++) {
        results[i].resize(dim);
    }
    // 5. 回到文件开头
    in.seekg(0, std::ios::beg);
    // 6. 只读取 num 条
    for (unsigned i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);               // 跳过 dim
        in.read((char*)results[i].data(), dim * 4);
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
float cal_recall_get_every_point_recall(std::vector<std::vector<unsigned>> results, std::vector<std::vector<unsigned>> true_data, unsigned num,
                 unsigned k,const char *every_points_recall_filename) {
    //统计进行模型预测时，模型调用情况
    //    std::ofstream csvFile("/home/extra_home/lxx23125236/ann-data/PGAS/query_NSSG_k_100_预测0.99准确率停止时各个数据点准确率情况.csv", std::ios::app);
    std::ofstream csvFile(every_points_recall_filename, std::ios::app);
    csvFile << "qid" << ","<< "r"<< "\n";
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
        csvFile << i << ","<< acc/k << "\n";
        //        csvFile << acc / k;
        //        csvFile << std::endl;
        mean_acc += acc / k;
    }
    return mean_acc / num;
}
float cal_low_recall_ratio(const std::vector<std::vector<unsigned>> &results,const std::vector<std::vector<unsigned>> &true_data,unsigned num,unsigned k,
        float target_recall) {

    unsigned low_cnt = 0;
    for (size_t i = 0; i < num; i++) {
        unsigned hit = 0;

        for (size_t j = 0; j < k; j++) {
            for (size_t m = 0; m < k; m++) {
                if (results[i][j] == true_data[i][m]) {
                    hit++;
                    break;
                }
            }
        }
        float recall_i = static_cast<float>(hit) / k;
        if (recall_i < target_recall) {
            low_cnt++;
        }
    }
    return static_cast<float>(low_cnt) / num;
}

// 一次性把单列 CSV 读成 vector<double>
inline std::vector<float> load_every_point_r(const std::string &csv_file)
{
    std::ifstream fin(csv_file);
    if (!fin) {
        throw std::runtime_error("cannot open csv file: " + csv_file);
    }
    std::vector<float> r_list;
    std::string line;
    // 1. 跳过表头
    if (!std::getline(fin, line)) {
        throw std::runtime_error("empty csv file: " + csv_file);
    }
    // 2. 逐行读取
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string qid_str, r_str;
        // 3. 读 qid
        if (!std::getline(ss, qid_str, ',')) continue;
        // 4. 读 r
        if (!std::getline(ss, r_str, ',')) continue;
        // 5. 转 float
        r_list.push_back(std::stof(r_str));
    }
    return r_list;
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

    unsigned L = 100;
    unsigned K = 100;
    unsigned query_num = 10000;

    float target_recall = 0.9f;

    std::string mode;          // train_data / test / get_train_CNS / ...
    std::string model_path;       // csv 或 model path
    std::string model_path_classification;       // csv 或 model path
    std::string out_put_path;       // csv 或 model path
    //Laet 方法参数
    unsigned laet_F=0;
    float laet_multiplier=1.0f;
    //Darth 方法参数
    unsigned ipi=500;
    unsigned mpi=100;

    unsigned stability_times = 0;
    unsigned stability_times_r1 = 0;
    std::string every_point_recall_path; // 仅 K=10 时使用
};

Args parse_args(int argc, char **argv) {
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

    // ===== 必须参数检查 =====
    auto require = [&](const std::string &k) {
        if (!kv.count(k))
            throw std::runtime_error("Missing required argument: --" + k);
    };
    //必填参数
    require("dataset");
    require("query_type");
//    require("L");
    require("K");
    require("query_num");
//    require("target_recall");
    require("mode");
//    require("io");

    std::string dataset = kv.at("dataset");
    std::string query_type = kv.at("query_type"); // train / test

    if (query_type != "train" && query_type != "test") {
        throw std::runtime_error(
                "Invalid --query_type, must be train or test"
        );
    }

    Args a;
    /* ========= 其余参数 ========= */
    if (kv.count("L")) a.L = std::stoul(kv.at("L"));
    if (kv.count("K")) a.K = std::stoul(kv.at("K"));
    if (kv.count("query_num")) a.query_num = std::stoul(kv.at("query_num"));
    if (kv.count("target_recall")) a.target_recall = std::stof(kv.at("target_recall"));
    if (kv.count("model_path")) a.model_path = kv.at("model_path");
    if (kv.count("model_path_classification")) a.model_path_classification = kv.at("model_path_classification");
    if (kv.count("out_put_path")) a.out_put_path = kv.at("out_put_path");

    if (kv.count("per_query_recall")) a.every_point_recall_path = kv.at("per_query_recall");
    if (kv.count("laet_multiplier")) a.laet_multiplier = std::stof(kv.at("laet_multiplier"));
//    if (kv.count("laet_multiplier")) a.laet_multiplier = std::stoul(kv.at("laet_multiplier"));
    if (kv.count("laet_F")) a.laet_F = std::stoul(kv.at("laet_F"));

    if (kv.count("ipi")) a.ipi = std::stoul(kv.at("ipi"));
    if (kv.count("mpi")) a.mpi = std::stoul(kv.at("mpi"));

    if (kv.count("stability_times")) a.stability_times = std::stoul(kv.at("stability_times"));
    if (kv.count("stability_times_r1")) a.stability_times_r1 = std::stoul(kv.at("stability_times_r1"));

    a.mode = kv.at("mode");

    /* ========= dataset 路径规则 ========= */
    if (dataset == "SIFT1M") {
        a.base_data = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_base.fvecs";
        a.graph_path = "/home/extra_home/lxx23125236/ali/NSG-data/sift_40_50_500.nsg";

        if (query_type == "train") {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_learn.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/SIFT1M/learn_groundtruth.ivecs";
        } else {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_query.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_groundtruth.ivecs";
        }

    } else if (dataset == "GIST1M") {
        a.base_data = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_base.fvecs";
        a.graph_path = "/home/extra_home/lxx23125236/ali/NSG-data/gist_60_70_500.nsg";

        if (query_type == "train") {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_learn.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_learn_groundtruth_50wan.ivecs";
        } else {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_query.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_groundtruth.ivecs";
        }

    } else if (dataset == "DEEP10M") {
        a.base_data = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M.fvecs";
        a.graph_path = "/home/extra_home/lxx23125236/ali/NSG-data/deep_60_70_500.nsg";

        if (query_type == "train") {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_learn_10wan_groundtruth_new.ivecs";
        } else {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_query_1wan.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_query_1wan_groundtruth.ivecs";
        }

    } else if (dataset == "GLOVE100") {
        a.base_data = "/home/extra_home/lxx23125236/ann-data/GLOVE100/base.1183514.fvecs";
        a.graph_path = "/home/extra_home/lxx23125236/ali/NSG-data/glove_60_70_500.nsg";

        if (query_type == "train") {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/GLOVE100/learn.100K.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/GLOVE100/learn.groundtruth.100K.k1000.ivecs";
        } else {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/GLOVE100/query.10K.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/GLOVE100/query.groundtruth.10K.k1000.ivecs";
        }

    }else if (dataset == "T2I1M") {
        a.base_data = "/home/extra_home/lxx23125236/ann-data/T2I1M/base.1B.fbin.fvecs";
        a.graph_path = "/home/extra_home/lxx23125236/ali/NSG-data/t2i_60_70_500.nsg";

        if (query_type == "train") {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/T2I1M/query.heldout.30K.train2w.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/T2I1M/gt100-heldout.30K.l2.train2w.ivecs";
        } else {
            a.query_data = "/home/extra_home/lxx23125236/ann-data/T2I1M/query.heldout.30K.query1w.fvecs";
            a.groundtruth_path = "/home/extra_home/lxx23125236/ann-data/T2I1M/gt100-heldout.30K.l2.query1w.ivecs";
        }
    }else {
        throw std::runtime_error("Unknown dataset: " + dataset);
    }
    return a;
}
////获取训练数据集达到要求准确率时的最小CNS
void run_train_min_CNS(efanna2e::IndexNSG &index, const Args &args, float *query, unsigned query_num, unsigned dim,efanna2e::Parameters &paras, const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));

    std::set<unsigned> visited;
    std::map<unsigned, float> L_acc;

    unsigned high = args.K;
    unsigned L_min = UINT_MAX;

    // 指数扩展
    while (true) {
        paras.Set<unsigned>("L_search", high);
        for (unsigned i = 0; i < query_num; i++) {
            index.SearchWithOptGraph(query + i * dim,args.K,paras,res[i].data());
        }
        float acc = cal_recall(res, groundtruth, query_num, args.K);

        visited.insert(high);
        L_acc[high] = acc;
        std::cout << "CNS=" << high<< " Recall=" << acc << std::endl;
        if (acc >= args.target_recall)
            break;
        high *= 2;
    }

    // 二分
    unsigned left = args.K;
    unsigned right = high;

    while (left <= right) {
        unsigned mid = left + (right - left) / 2;
        float acc;

        if (visited.count(mid)) {
            acc = L_acc[mid];
        } else {
            paras.Set<unsigned>("L_search", mid);
            for (unsigned i = 0; i < query_num; i++) {
                index.SearchWithOptGraph(query + i * dim,args.K,paras,res[i].data());
            }
            acc = cal_recall(res, groundtruth, query_num, args.K);
            visited.insert(mid);
            L_acc[mid] = acc;
        }

        if (acc >= args.target_recall) {
            L_min = std::min(L_min, mid);
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    std::cout << "训练集最小 CNS = "<< L_min << std::endl;
}
//REM基础搜索方法 baseline
void REM_test(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
              const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
//        index.SearchWithOptGraph(query + i * dim,args.K,i,paras,res[i].data());
        index.SearchWithOptGraph_get_every_points_recall(query + i * dim,args.K,i,paras,res[i].data(),args.out_put_path.c_str());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << std::endl;

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN Recall = "<< recall << std::endl;
}
//获取Raet预测准确率方法训练数据集
void run_Laet_Darth_train_data(
        efanna2e::IndexNSG &index,
        const Args &args,
        float *query,
        unsigned query_num,
        unsigned dim,
        efanna2e::Parameters &paras) {
    //写入表头
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid," << "step,"<< "dists,"<< "inserts," << "visited_points," << "duration,"
    << "first_nn_dist,"<< "nn_dist,"<< "furthest_dist,"
    << "nn10_dist,"<< "nn_to_first,"<< "nn10_to_first,"
    << "mean_distance,"<<"variance_of_distances,"
    <<"percentile_25,"<<"percentile_50,"<<"percentile_75,"<< "r\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Laet_Darth_train_data(query + i * dim, args.K, paras, res[i].data(), i,args.out_put_path.c_str());
    }
    std::cout << "特征与标签已收集，CSV 路径: " << args.out_put_path.c_str() << std::endl;
}
//Laet搜索方法
void run_laet_test(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
              const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predicted_CNS,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_laet_test(query + i * dim,args.K,i,paras,res[i].data(),args.laet_F,args.laet_multiplier,booster,args.out_put_path.c_str());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall <<"\n";

    float low_target_recall = cal_low_recall_ratio(res, groundtruth, query_num, args.K,args.target_recall);
    std::cout << "低于要求准确率:"<< args.target_recall << "的占比为" << low_target_recall << std::endl;
}
void run_Darth_test_time(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predict_recall,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Darth_test_time(query + i * dim,args.K,paras,res[i].data(),i,args.ipi,args.mpi,args.target_recall,booster,args.out_put_path.c_str());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";
    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall <<"\n";
}

void run_Darth_test_metrics(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
                         const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    unsigned success_stop_count=0;
    unsigned stop_count=0;
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Darth_test_metrics(query + i * dim,args.K,paras,res[i].data(),i,args.ipi,args.mpi,args.target_recall,booster,success_stop_count,stop_count);
    }
    //模型指标
    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";
    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall <<"\n";

    float low_target_recall = cal_low_recall_ratio(res, groundtruth, query_num, args.K,args.target_recall);
    std::cout << "低于要求准确率:"<< args.target_recall << "的占比为" << low_target_recall <<"\n";

    //早停率：早停（预测准确率>target_recall）/ 总数据点数量
    std::cout << "平均每个数据点上的早停率:"<< static_cast<double>(stop_count)/query_num <<"\n";

    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 总数据点数量
    std::cout << "平均每个数据点上的成功早停率:"<< static_cast<double>(success_stop_count)/query_num <<"\n";

    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 预测次数
    std::cout << "平均每次预测的成功早停率:"<< static_cast<double>(success_stop_count)/model_pre_count << std::endl;
    // ===== 写入 JSON 结果  覆盖写 =====
    {
        std::ofstream ofs(args.out_put_path);
        if (!ofs.is_open()) {
            std::cerr << "无法打开输出文件: " << args.out_put_path << std::endl;
            return;
        }
        ofs << std::fixed << std::setprecision(6);
        ofs << "{\n";
        ofs << "  \"K\": " << args.K << ",\n";
        ofs << "  \"target_recall\": " << args.target_recall << ",\n";
        ofs << "  \"metrics\": {\n";
        ofs << "    \"model_predict_count\": " << model_pre_count << ",\n";
        ofs << "    \"model_predict_time_sec\": " << sec << ",\n";
        ofs << "    \"recall\": " << recall << ",\n";
        ofs << "    \"low_target_recall_ratio\": " << low_target_recall << ",\n";
        ofs << "    \"success_early_stop_per_query\": "<< static_cast<double>(success_stop_count) / query_num << ",\n";
        ofs << "    \"success_early_stop_per_prediction\": "<< static_cast<double>(success_stop_count) / model_pre_count << "\n";
        ofs << "  }\n";
        ofs << "}\n";
        ofs.close();
    }
}
//Raet预测准确率方法-获取训练集达到要求准确率的平均稳定次数
void run_our_Raet_get_stablize_count(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    int all_stability_counts=0;
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<< "step,"<<"stability,"<< "r\n";
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_stablize(query + i * dim,args.K,paras,res[i].data(),args.target_recall,i,all_stability_counts,args.out_put_path.c_str());
    }
    std::cout << "训练集平均稳定次数为: "<< all_stability_counts/query_num << std::endl;
}

//获取Raet预测准确率方法训练数据集，只收集大于0.9准确率时的数据，每个数据点在每个准确率下行数不限，准确率为1限制为最多20行
void run_our_Raet_train_data(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras) {
    //写入表头
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<< "step,"<< "dists,"<< "inserts,"<<"visited_points,"
           << "first_nn_dist,"<< "nn_dist,"<< "furthest_dist,"
           << "nn10_dist,"<< "nn_to_first,"<< "nn10_to_first,"
           << "mean_distance,"<<"variance_of_distances,"
           <<"percentile_25,"<<"percentile_50,"<<"percentile_75,"<< "r\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Raet_recall_train_data(query + i * dim, args.K, paras, res[i].data(), i,args.out_put_path.c_str());
    }
    std::cout << "Raet_recall特征与标签已收集，CSV 路径: " << args.out_put_path.c_str() << std::endl;
}
//Raet预测准确率方法 测试数据的模型预测
void run_our_Raet_test(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,const std::vector<std::vector<unsigned>> &groundtruth) {

    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predict_recall,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);

    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        //正常方法
        index.SearchWithOptGraph_pre_recall_lessPre(query + i * dim,args.K,i,paras,res[i].data(),args.target_recall,args.stability_times,args.stability_times_r1,booster,args.out_put_path.c_str());
        //
//        index.SearchWithOptGraph_pre_recall_lessPre_test(query + i * dim,args.K,i,paras,res[i].data(),args.target_recall,args.stability_times,args.stability_times_r1,booster,args.out_put_path.c_str());

//            index.SearchWithOptGraph_pre_recall_lessPre_darthFeature(query + i * dim,args.K,i,paras,res[i].data(),args.target_recall,args.stability_times,args.stability_times_r1,booster);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

//        size_t ns = index.GetTimeSpendModel();           // 纳秒
//        double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
//        std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";
//        size_t model_pre_count = index.GetModelPreCount();
//        std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall<< std::endl;
}
//选择器方法获取召回率预测标签
void run_our_Raet_selector_label(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,const std::vector<std::vector<unsigned>> &groundtruth) {

    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predict_recall,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);

    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        //正常方法
        index.SearchWithOptGraph_pre_recall_seletor_label(query + i * dim,args.K,i,paras,res[i].data(),args.target_recall,args.stability_times,args.stability_times_r1,booster,args.out_put_path.c_str());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";
    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall<< std::endl;
}
void run_our_Raet_test_metrics(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
                               const std::vector<std::vector<unsigned>> &groundtruth) {

    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    unsigned success_stop_count=0;
    unsigned stop_count=0;
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Raet_test_metrics(query + i * dim,args.K,paras,res[i].data(),i,args.target_recall,args.stability_times,args.stability_times_r1,booster,success_stop_count,stop_count,args.out_put_path.c_str());
    }
    //模型指标
    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";
    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall <<"\n";

    float low_target_recall = cal_low_recall_ratio(res, groundtruth, query_num, args.K,args.target_recall);
    std::cout << "低于要求准确率:"<< args.target_recall << "的占比为" << low_target_recall <<"\n";

    //早停率：早停（预测准确率>target_recall）/ 总数据点数量
    std::cout << "平均每个数据点上的早停率:"<< static_cast<double>(stop_count)/query_num <<"\n";

    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 总数据点数量
    std::cout << "平均每个数据点上的成功早停率:"<< static_cast<double>(success_stop_count)/query_num <<"\n";

    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 总数据点数量
    std::cout << "平均每次预测的成功早停率:"<< static_cast<double>(success_stop_count)/model_pre_count << std::endl;
    // ===== 写入 JSON 结果 =====
//    {
//        std::ofstream ofs(args.out_put_path);
//        if (!ofs.is_open()) {
//            std::cerr << "无法打开输出文件: " << args.out_put_path << std::endl;
//            return;
//        }
//        ofs << std::fixed << std::setprecision(6);
//        ofs << "{\n";
//        ofs << "  \"K\": " << args.K << ",\n";
//        ofs << "  \"target_recall\": " << args.target_recall << ",\n";
//        ofs << "  \"metrics\": {\n";
//        ofs << "    \"model_predict_count\": " << model_pre_count << ",\n";
//        ofs << "    \"model_predict_time_sec\": " << sec << ",\n";
//        ofs << "    \"recall\": " << recall << ",\n";
//        ofs << "    \"low_target_recall_ratio\": " << low_target_recall << ",\n";
//        ofs << "    \"success_early_stop_per_query\": "<< static_cast<double>(success_stop_count) / query_num << ",\n";
//        ofs << "    \"success_early_stop_per_prediction\": "<< static_cast<double>(success_stop_count) / model_pre_count << "\n";
//        ofs << "  }\n";
//        ofs << "}\n";
//        ofs.close();
//    }

}
//获取数据集达到平均准确率时，每个数据点的准确率
void get_every_point_recall(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
              const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph(query + i * dim,args.K,paras,res[i].data());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << std::endl;

    float recall = cal_recall_get_every_point_recall(res, groundtruth, query_num, args.K,args.out_put_path.c_str());
    std::cout << args.K << " NN Recall = "<< recall << std::endl;
}
//todo 获取Raet预测CNS方法训练数据集
void run_our_Raet_CNS_train_data(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras) {
    //写入表头
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<< "step,"<< "dists,"<< "inserts,"
           << "first_nn_dist,"<< "nn_dist,"<< "furthest_dist,"
           << "mean_distance,"<<"variance_of_distances,"
           <<"avg_hop,"<<"avg_indegree,"<<"density,"<<"entry_query_dist_ratio,"<<"dist_distribution_entropy,"
           <<"skewness,"<<"kurtosis,"<< "energy,"
           <<"cosine_mean,"<<"cosine_variance,"<< "cosine_direction_entropy\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Raet_CNS_train_data(query + i * dim, args.K, paras, res[i].data(), args.target_recall, i,
                                            args.out_put_path.c_str());
    }
    std::cout << "Raet_CNS特征与标签已收集，CSV 路径: " << args.out_put_path << std::endl;
}



//预测CNS的标签，获取每个数据点达到要求准确率的最小CNS
void run_our_Raet_CNS_find_every_point_min_L(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream output(args.out_put_path);
    if (!output.is_open())
        throw std::runtime_error("无法创建输出文件");
    // ===== 表头 =====
    output << "qid,L_min\n";
    std::vector<unsigned> res(args.K);
    unsigned avg_CNS = static_cast<unsigned>(args.L);
    for (unsigned i = 0; i < query_num; i++) {
        paras.Set<unsigned>("L_search", avg_CNS);
        index.SearchWithOptGraph(query + i * dim,args.K,paras,res.data());
        float acc = cal_recall_one_point(res, groundtruth[i], args.K);
        if (acc < args.target_recall) {
            // 理论上不应发生
            output << i << "," << avg_CNS << "\n";
            continue;
        }
        unsigned left = args.K;
        unsigned right = avg_CNS;
        unsigned L_min = avg_CNS;

        while (left <= right) {
            unsigned mid = left + (right - left) / 2;

            paras.Set<unsigned>("L_search", mid);
            index.SearchWithOptGraph(query + i * dim, args.K, paras, res.data());
            float mid_acc = cal_recall_one_point(res, groundtruth[i], args.K);

            if (mid_acc >= args.target_recall) {
                L_min = mid;
                right = mid - 1;       // try smaller L
            } else {
                left = mid + 1;        // L too small
            }
        }
        output << i << "," << L_min << "\n";
    }
    std::cout << "逐 query 最小 CNS 已保存至: "<< args.out_put_path << std::endl;
}
//Raet预测准确率方法 测试数据的模型预测
void run_our_Raet_CNS_test_regression(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predicted_CNS,"<<"r_actual,"<< "search_time\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);

    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
//            index.SearchWithOptGraph_our_Raet_CNS_test_regression(query + i * dim,args.K,paras,res[i].data(), i ,args.target_recall,booster);
        index.SearchWithOptGraph_our_Raet_CNS_test_regression_Noquery_Noske_Nodensty(query + i * dim,args.K,paras,res[i].data(), args.target_recall, i ,booster,args.out_put_path.c_str());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";
    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall<< std::endl;

}
void run_our_Raet_CNS_test_regression_metrics(efanna2e::IndexNSG &index,const Args &args,float *query,unsigned query_num,unsigned dim,efanna2e::Parameters &paras,
                               const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    unsigned success_stop_count=0;
    //加载模型
    int out_iterations;
    BoosterHandle booster;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_Raet_test_regression_metrics(query + i * dim, args.K, paras, res[i].data(), i,
                                                              args.target_recall, booster,
                                                              success_stop_count);
    }
    //模型指标
    size_t model_pre_count = index.GetModelPreCount();
    std::cout << "模型预测次数:"<< model_pre_count <<"\n";
    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):"<< std::fixed << std::setprecision(6)<< sec << " seconds\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = "<< recall <<"\n";

    float low_target_recall = cal_low_recall_ratio(res, groundtruth, query_num, args.K,args.target_recall);
    std::cout << "低于要求准确率:"<< args.target_recall << "的占比为" << low_target_recall <<"\n";

//    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 总数据点数量
//    std::cout << "平均每个数据点上的成功早停率:"<< static_cast<double>(success_stop_count)/query_num <<"\n";
//
//    //成功早停率：成功早停（真实准确率>target_recall && 预测准确率>target_recall）/ 总数据点数量
//    std::cout << "平均每次预测的成功早停率:"<< static_cast<double>(success_stop_count)/model_pre_count << std::endl;
    // ===== 写入 JSON 结果 =====
    {
        std::ofstream ofs(args.out_put_path);
        if (!ofs.is_open()) {
            std::cerr << "无法打开输出文件: " << args.out_put_path << std::endl;
            return;
        }
        ofs << std::fixed << std::setprecision(6);
        ofs << "{\n";
        ofs << "  \"K\": " << args.K << ",\n";
        ofs << "  \"target_recall\": " << args.target_recall << ",\n";
        ofs << "  \"metrics\": {\n";
        ofs << "    \"model_predict_count\": " << model_pre_count << ",\n";
        ofs << "    \"model_predict_time_sec\": " << sec << ",\n";
        ofs << "    \"recall\": " << recall << ",\n";
        ofs << "    \"low_target_recall_ratio\": " << low_target_recall << ",\n";
        ofs << "    \"success_early_stop_per_query\": "<< static_cast<double>(success_stop_count) / query_num << ",\n";
        ofs << "    \"success_early_stop_per_prediction\": "<< static_cast<double>(success_stop_count) / model_pre_count << "\n";
        ofs << "  }\n";
        ofs << "}\n";
        ofs.close();
    }

}
//Raet预测准确率方法 测试数据的模型预测
void run_our_Raet_CNS_test_classification(efanna2e::IndexNSG &index, const Args &args, float *query,
                                                     unsigned query_num, unsigned dim, efanna2e::Parameters &paras,
                                                     const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster_classification;
    int result_1=LGBM_BoosterCreateFromModelfile(args.model_path_classification.c_str(), &out_iterations, &booster_classification);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_our_Raet_CNS_test_classification(query + i * dim, args.K, paras, res[i].data(), i,
                                                                             args.target_recall, booster_classification);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):" << std::fixed << std::setprecision(6) << sec << " seconds\n";
//        size_t model_pre_count = index.GetModelPreCount();
//        std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = " << recall << std::endl;

}
//Raet预测准确率方法 测试数据的模型预测
void run_our_Raet_CNS_test_regression_classification(efanna2e::IndexNSG &index, const Args &args, float *query,
                                                     unsigned query_num, unsigned dim, efanna2e::Parameters &paras,
                                                     const std::vector<std::vector<unsigned>> &groundtruth) {
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster_regression;
    int result = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster_regression);
    BoosterHandle booster_classification;
    int result_1=LGBM_BoosterCreateFromModelfile(args.model_path_classification.c_str(), &out_iterations, &booster_classification);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_our_Raet_CNS_test_regerssion_classification(query + i * dim, args.K, paras, res[i].data(), i,
                                                                  args.target_recall, booster_regression,booster_classification);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):" << std::fixed << std::setprecision(6) << sec << " seconds\n";
//        size_t model_pre_count = index.GetModelPreCount();
//        std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = " << recall << std::endl;

}
//Raet预测准确率方法 测试数据的模型预测
void run_our_Raet_CNS_selector(efanna2e::IndexNSG &index, const Args &args, float *query,
                                                     unsigned query_num, unsigned dim, efanna2e::Parameters &paras,
                                                     const std::vector<std::vector<unsigned>> &groundtruth) {
    std::ofstream header(args.out_put_path, std::ios::out);
    header << "qid,"<<"predict_recall,"<<"r_actual,"<<"search_time,"<< "selector\n";
    header.close();
    std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(args.K));
    //加载模型
    int out_iterations;
    BoosterHandle booster_recall;
    int result_recall = LGBM_BoosterCreateFromModelfile(args.model_path.c_str(), &out_iterations, &booster_recall);
    BoosterHandle booster_CNS;
    int result_CNS=LGBM_BoosterCreateFromModelfile(args.model_path_classification.c_str(), &out_iterations, &booster_CNS);
    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < query_num; i++) {
        index.SearchWithOptGraph_our_Raet_CNS_selector(query + i * dim, args.K, paras, res[i].data(), i,
                                                                             args.target_recall, args.stability_times,args.stability_times_r1,booster_recall,booster_CNS,args.out_put_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "总搜索时间:" << diff.count() << "\n";

    size_t ns = index.GetTimeSpendModel();           // 纳秒
    double sec = static_cast<double>(ns) / 1'000'000'000.0; // 转换为秒
    std::cout << "总模型预测时间(s):" << std::fixed << std::setprecision(6) << sec << " seconds\n";
//        size_t model_pre_count = index.GetModelPreCount();
//        std::cout << "模型预测次数:"<< model_pre_count <<"\n";

    float recall = cal_recall(res, groundtruth, query_num, args.K);
    std::cout << args.K << " NN 准确率 = " << recall << std::endl;

}
int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    // === 数据加载 ===
    float *data_load = nullptr;
    unsigned points_num, dim;
    load_data(args.base_data.c_str(), data_load, points_num, dim);

    float *query_load = nullptr;
    unsigned query_num, query_dim;
    load_train_or_test_data(args.query_data.c_str(), query_load, query_num, query_dim,args.query_num);
//    load_data(args.query_data.c_str(), query_load, query_num, query_dim);
    assert(dim == query_dim);

    // === Groundtruth ===
    std::vector<std::vector<unsigned>> true_load;
    unsigned true_num, true_dim;
    load_ivecs_data(args.groundtruth_path.c_str(), true_load, true_num, true_dim, args.query_num);

    /* ========= 索引初始化 ========= */
    efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
    index.Load(args.graph_path.c_str());
    index.OptimizeGraph(data_load);

    index.InitTimeSpendModel();
    index.InitDistCount();
    index.InitModelPreCount();
    //获取每个数据点的入度值,同时，初始化跳数
    index.GetIndegree();
    //归一化,方便进行CNS预测时，计算余弦相似度
    index.Normalization_db_and_query(data_load,query_load,points_num,query_num,dim);

    index.load_ivecs_data(args.groundtruth_path.c_str(), true_num, true_dim,args.query_num);

    /* ========= 搜索参数 ========= */
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", args.L);
    paras.Set<unsigned>("P_search", args.L);

    /* ========= 模式分发 ========= */
    //获取CNS方法的，训练数据 达到要求准确率的最小CNS
    if (args.mode == "get_min_CNS_of_train_data") {
        run_train_min_CNS(index, args,query_load, query_num, dim,paras,true_load);
    }

    //基础搜索方法，baseli REM
    else if (args.mode == "REM_test") {
        REM_test(index, args,query_load, query_num, dim,paras,true_load);
    }

    //Laet方法获取训练数据，
    // 该方法获取的训练数据，是准确率从0开始，每个数据点的每一个准确率最多有10行数据，每个数据点的准确率为1最多有20行数据
    else if (args.mode == "Laet_Darth_train_data") {
        run_Laet_Darth_train_data(index, args,query_load, query_num, dim,paras);
    }
    //Laet方法获取训练数据
    else if (args.mode == "Laet_test") {
        run_laet_test(index, args,query_load, query_num, dim,paras,true_load);
    }
    //只计算时间
    else if (args.mode == "Darth_test_time") {
        run_Darth_test_time(index, args,query_load, query_num, dim,paras,true_load);
    }
    //统计Darth的所有指标
    else if (args.mode == "Darth_test_metrics") {
        run_Darth_test_metrics(index, args,query_load, query_num, dim,paras,true_load);
    }

    //预测准确率方法，获取训练集达到要求准确率的平均距离计算次数，现在是将稳定次数输出到文件中
    else if (args.mode == "our_Raet_recall_get_StablizeCount") {
        run_our_Raet_get_stablize_count(index, args,query_load, query_num, dim,paras);
    }
    //该方法，获取的是，准确率大于0.9的训练数据，其中每一个数据点准确率为1的行数最多有20行。
    // 数据量太大，内存不足，无法训练模型，只能取0.95准确率之后的数据点了
    else if (args.mode == "our_Raet_train_data") {
        run_our_Raet_train_data(index, args,query_load, query_num, dim,paras);
    }
    //k=10,获取平均达到要求准确率时，每个数据点达到的准确率
    else if (args.mode == "get_every_point_recall") {
        get_every_point_recall(index, args,query_load, query_num, dim,paras,true_load);
    }

    //预测准确率方法的搜索
    else if (args.mode == "our_Raet_test") {
        run_our_Raet_test(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_selector_label") {
        run_our_Raet_selector_label(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_test_metrics") {
        run_our_Raet_test_metrics(index, args,query_load, query_num, dim,paras,true_load);
    }
    else if (args.mode == "our_Raet_CNS_train_data") {
        run_our_Raet_CNS_train_data(index, args,query_load, query_num, dim,paras);
    }
        //获取训练数据集中 ，每个数据点达到要求准确率需要的最小CNS, 是预测CNS的标签
    else if (args.mode == "our_Raet_CNS_train_label") {
        run_our_Raet_CNS_find_every_point_min_L(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_CNS_test_regression") {
        run_our_Raet_CNS_test_regression(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_CNS_test_regression_metrics") {
        run_our_Raet_CNS_test_regression_metrics(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_CNS_test_regression_classification") {
        run_our_Raet_CNS_test_regression_classification(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_CNS_test_classification") {
        run_our_Raet_CNS_test_classification(index, args,query_load, query_num, dim,paras, true_load);
    }
    else if (args.mode == "our_Raet_CNS_selector") {
        run_our_Raet_CNS_selector(index, args,query_load, query_num, dim,paras, true_load);
    }
//    else if (args.mode == "our_Raet_CNS_test_classification_metrics") {
//        run_our_Raet_CNS_test_classification_metrics(index, args,query_load, query_num, dim,paras, true_load);
//    }
    else {
        throw std::runtime_error("Unknown mode: " + args.mode);
    }
    return 0;
}

