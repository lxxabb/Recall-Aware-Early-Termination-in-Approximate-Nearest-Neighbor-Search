import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.special import expit  # Sigmoid 函数
import json

# ================= 路径配置 =================
DATA_ROOT = r"D:\1-work-23281191\大创\1-21\data"
MODEL_SAVE_DIR = r"D:\1-work-23281191\121_models"


# ===========================================

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    # 验证集特征
    return os.path.join(DATA_ROOT, "val", f"efS{efS}_qs{num_queries}.txt")


def get_validation_label_dataset_name(ds_name, k, target_recall, query_num):
    # 验证集标签
    return os.path.join(DATA_ROOT, "val", f"min_ef_per_query_s{query_num}_{target_recall}.csv")


def get_CNS(ds_name, target_recall):
    if ds_name == "GLOVE100":
        if target_recall == 0.99: return 247
        if target_recall == 0.98: return 116
    return 100


def main():
    # 只跑 GLOVE100
    all_datasets = ["GLOVE100"]
    dataset_params = {
        "GLOVE100": {"M": 16, "efC": 500, "efS": 500, "d": 100, "query_num": 10000},
    }

    test_queries_num = 10000
    all_k_values = [100]
    all_li = [1]
    target_recalls = [0.98]

    # 特征列表
    input_features = ["step", "dists", "inserts", "first_nn_dist", "nn_dist", "furthest_dist",
                      "avg_dist", "variance", "avg_hop", "density", "entry_query_dist_ratio",
                      "dist_distribution_entropy", "skewness", "kurtosis", "energy",
                      "cosine_mean", "cosine_variance", "cosine_direction_entropy"]

    columns_to_load = ["elaps_ms", "qid", "r", "feats_collect_time_ms"] + input_features

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]

        print(f"--- 开始评估 Logit 模型性能 (Dataset: {ds_name}) ---")

        for k in all_k_values:
            li = 1
            data_path = get_dataset_name(M, efC, efS, test_queries_num, ds_name, k, li)

            if not os.path.exists(data_path):
                print(f"❌ 找不到特征文件: {data_path}")
                continue

            query_data_all = pd.read_csv(data_path, usecols=columns_to_load)

            for target_recall in target_recalls:
                CNS = get_CNS(ds_name, target_recall)
                if CNS == k: continue

                # 定义范围 [a, b]
                val_a = k
                val_b = CNS

                print(f"\n>>> 目标召回率: {target_recall}")
                print(f"    范围约束: [{val_a}, {val_b}]")

                label_path = get_validation_label_dataset_name(ds_name, k, target_recall, test_queries_num)
                if not os.path.exists(label_path):
                    print(f"❌ 找不到标签文件: {label_path}")
                    continue

                min_cns_df = pd.read_csv(label_path).set_index("qid")

                for li in all_li:
                    data_all = query_data_all[query_data_all["dists"] % li == 0].copy()
                    data_all = data_all.merge(min_cns_df[["min_efSearch"]], left_on="qid", right_index=True,
                                              how="inner")

                    X_test = data_all[input_features].values
                    y_true = data_all["min_efSearch"].values
                    # 标签截断 (为了公平对比)
                    y_true = np.clip(y_true, val_a, val_b)

                    # 加载 Logit 模型
                    model_file = os.path.join(MODEL_SAVE_DIR, ds_name, f"model_{target_recall}_logit.txt")
                    if not os.path.exists(model_file):
                        print(f"❌ Logit模型文件不存在: {model_file}")
                        print("   请先运行 3-1-logit-训练.py")
                        continue

                    model = lgb.Booster(model_file=model_file)

                    # 1. 预测 Logits (z)
                    z_pred = model.predict(X_test)

                    # 2. 还原为步数 (L)
                    # 公式: L = a + (b-a) * Sigmoid(z)
                    # 注意：体检时我们通常不乘 Gamma (Gamma=1.0)，只看模型原本准不准
                    y_pred = val_a + (val_b - val_a) * expit(z_pred)

                    # 3. 评估指标 (对比真实步数)
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    print(f"    [体检报告 - Logit版]")
                    print(f"    平均绝对误差 (MAE): {mae:.4f}")
                    print(f"      -> 含义: 模型还原后的步数平均偏离真实值 {mae:.2f} 步")
                    print(f"    均方误差 (MSE)   : {mse:.4f}")
                    print(f"    决定系数 (R2)    : {r2:.4f}")

                    # 统计偏离
                    diff = y_pred - y_true
                    over_estimate = np.mean(diff > 0)
                    print(f"    倾向性分析:")
                    print(f"      - 偏保守(预测 > 真实): {over_estimate * 100:.1f}%")
                    print(f"      - 偏激进(预测 < 真实): {(1 - over_estimate) * 100:.1f}%")

    print("\n--- 评估结束 ---")


if __name__ == "__main__":
    main()