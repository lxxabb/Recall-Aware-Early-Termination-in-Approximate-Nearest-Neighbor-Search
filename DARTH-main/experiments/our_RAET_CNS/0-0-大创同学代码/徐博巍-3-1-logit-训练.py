import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit  # 引入反Sigmoid函数
import json

# ================= 路径配置 =================
DATA_ROOT = r"D:\1-work-23281191\大创\1-21\data"
MODEL_SAVE_DIR = r"D:\1-work-23281191\121_models"


# ===========================================

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    # 训练集特征路径
    return os.path.join(DATA_ROOT, "train", f"efS{efS}_qs{num_queries}.txt")


def get_label_dataset_name(ds_name, k, target_recall, num_queries):
    # 训练集标签路径
    return os.path.join(DATA_ROOT, "train", f"min_ef_per_query_{target_recall}.csv")


def get_CNS(ds_name, target_recall):
    if ds_name == "GLOVE100":
        if target_recall == 0.99: return 247
        if target_recall == 0.98: return 116
    return 100


def main():
    SEED = 42
    n_estimators = 100

    # 特征定义 (保持不变)
    qid_feat = ["qid"]
    index_metric_feats = ["step", "dists", "inserts"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance", "avg_hop", "density", "entry_query_dist_ratio",
                            "dist_distribution_entropy", "skewness", "kurtosis", "energy"]
    cosine_feats = ["cosine_mean", "cosine_variance", "cosine_direction_entropy"]
    all_feats = qid_feat + index_metric_feats + neighbor_distances_feats + neighbor_stats_feats + cosine_feats
    columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

    model_conf = {
        "raet-cns": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose=-1),
    }

    dataset_params = {"GLOVE100": {"M": 16, "efC": 500, "efS": 500, "d": 100}}
    training_queries_num = [10000]
    all_k_values = [100]
    all_datasets = ["GLOVE100"]
    all_li = [1]
    target_recalls = [0.98]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]

        print(f"Starting Logit Training for {ds_name}...")

        for k in all_k_values:
            li = 1
            data_path = get_dataset_name(M, efC, efS, 10000, ds_name, k, li)
            all_queries_pd = pd.read_csv(data_path, usecols=columns_to_load)

            for s in training_queries_num:
                for target_recall in target_recalls:
                    CNS = get_CNS(ds_name, target_recall)
                    if CNS == k: continue

                    # 定义范围 [a, b] (对应 PDF)
                    val_a = k  # 100
                    val_b = CNS  # 116

                    print(f"准确率为 {target_recall} | 范围约束: [{val_a}, {val_b}]")

                    query_data_all = all_queries_pd[all_queries_pd["qid"] < s]
                    label_path = get_label_dataset_name(ds_name, k, target_recall, 10000)
                    min_cns_df = pd.read_csv(label_path).set_index("qid")

                    data_all = query_data_all[query_data_all["dists"] % li == 0].merge(
                        min_cns_df[["min_efSearch"]], left_on="qid", right_index=True)

                    y_raw = data_all["min_efSearch"].values

                    # ======================================================
                    # 【核心修改】Logit 变换 (完全符合 PDF 方案二训练要求)
                    # ======================================================
                    # 1. 硬截断到合法范围，防止数值越界
                    y_raw = np.clip(y_raw, val_a, val_b)

                    # 2. 归一化到 (0, 1) 之间
                    # eps 是为了防止 logit(0) 或 logit(1) 产生无穷大
                    eps = 1e-4
                    y_norm = (y_raw - val_a) / (val_b - val_a)
                    y_norm = np.clip(y_norm, eps, 1 - eps)

                    # 3. Logit 变换: z = ln(p / (1-p))
                    # 这就是模型要学习的目标！(无界数值)
                    y_train = logit(y_norm)

                    input_features = ["step", "dists", "inserts", "first_nn_dist", "nn_dist", "furthest_dist",
                                      "avg_dist", "variance", "avg_hop", "density", "entry_query_dist_ratio",
                                      "dist_distribution_entropy", "skewness", "kurtosis", "energy",
                                      "cosine_mean", "cosine_variance", "cosine_direction_entropy"]

                    X_train = data_all[input_features].values

                    model = model_conf["raet-cns"]
                    model.fit(X_train, y_train)

                    # 保存为 _logit 模型，以免覆盖旧模型
                    model_file = os.path.join(MODEL_SAVE_DIR, ds_name, f"model_{target_recall}_logit.txt")
                    os.makedirs(os.path.dirname(model_file), exist_ok=True)
                    model.booster_.save_model(model_file)
                    print(f"✅ Logit 模型保存到: {model_file}")

    print(f"Finished training.")


if __name__ == "__main__":
    main()