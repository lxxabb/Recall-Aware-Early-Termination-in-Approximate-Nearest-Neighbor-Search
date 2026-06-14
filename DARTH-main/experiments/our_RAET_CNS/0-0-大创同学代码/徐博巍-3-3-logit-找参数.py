import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error

# ================= 路径配置 =================
DATA_ROOT = r"D:\1-work-23281191\大创\1-21\data"
MODEL_SAVE_DIR = r"D:\1-work-23281191\121_models"


# ===========================================

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return os.path.join(DATA_ROOT, "val", f"efS{efS}_qs{num_queries}.txt")


def get_validation_label_dataset_name(ds_name, k, target_recall, query_num):
    return os.path.join(DATA_ROOT, "val", f"min_ef_per_query_s{query_num}_{target_recall}.csv")


def get_CNS(ds_name, target_recall):
    if ds_name == "GLOVE100":
        if target_recall == 0.99: return 247
        if target_recall == 0.98: return 116
    return 100


def objective_function(k, z_pred, y_true, val_a, val_b):
    # 标准 MSE：寻找数据的“重心”
    y_hat = val_a + (val_b - val_a) * expit(k * z_pred)
    mse = mean_squared_error(y_true, y_hat)
    return mse


def main():
    ds_name = "GLOVE100"
    target_recalls = [0.98]
    test_queries_num = 10000
    k_val = 100

    input_features = ["step", "dists", "inserts", "first_nn_dist", "nn_dist", "furthest_dist",
                      "avg_dist", "variance", "avg_hop", "density", "entry_query_dist_ratio",
                      "dist_distribution_entropy", "skewness", "kurtosis", "energy",
                      "cosine_mean", "cosine_variance", "cosine_direction_entropy"]
    columns_to_load = ["elaps_ms", "qid", "r", "feats_collect_time_ms"] + input_features

    data_path = get_dataset_name(16, 500, 500, test_queries_num, ds_name, 100, 1)
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到特征文件: {data_path}")
        return

    print(f"正在加载特征文件: {data_path}")
    query_data_all = pd.read_csv(data_path, usecols=columns_to_load)

    for target_recall in target_recalls:
        CNS = get_CNS(ds_name, target_recall)
        val_a = k_val
        val_b = CNS

        print(f"\n{'=' * 80}")
        print(f"开始 Stage 2 参数寻优 (目标召回率: {target_recall})")
        print(f"{'=' * 80}")

        label_path = get_validation_label_dataset_name(ds_name, k_val, target_recall, test_queries_num)
        if not os.path.exists(label_path):
            print(f"❌ 错误：找不到标签文件: {label_path}")
            continue

        min_cns_df = pd.read_csv(label_path).set_index("qid")
        data_all = query_data_all.merge(min_cns_df[["min_efSearch"]], left_on="qid", right_index=True)

        X_val = data_all[input_features].values
        y_true = data_all["min_efSearch"].values

        model_file = os.path.join(MODEL_SAVE_DIR, ds_name, f"model_{target_recall}_logit.txt")
        if not os.path.exists(model_file):
            print(f"❌ 错误：找不到 Logit 模型文件: {model_file}")
            continue

        model = lgb.Booster(model_file=model_file)
        z_pred = model.predict(X_val)

        # ---------------------------------------------------------
        # 步骤 1: 寻找最佳斜率 k (使用标准 MSE 确保模型稳健)
        # ---------------------------------------------------------
        print(">>> [步骤 1] 正在计算最佳分布曲线 (优化 k 值)...")
        result = minimize_scalar(
            fun=objective_function,
            bounds=(0.1, 5.0),
            args=(z_pred, y_true, val_a, val_b),
            method='bounded'
        )

        best_k = result.x

        # 计算“裸”预测值 (Bias=0 时)
        y_final_raw = val_a + (val_b - val_a) * expit(best_k * z_pred)
        y_final_base = np.ceil(y_final_raw).astype(int)

        print(f"    - 最佳 k 值: {best_k:.5f}")
        print(f"    - 原始中心覆盖率: {np.mean(y_final_base >= y_true) * 100:.2f}% (预期约 50%)")

        # ---------------------------------------------------------
        # 步骤 2: 智能优化安全系数 (Bias)
        # ---------------------------------------------------------
        print("\n>>> [步骤 2] 正在求解最小代价安全系数 (Optimal Bias)...")

        best_bias = 0
        final_cov = 0

        # 自动扫描：找到满足 target_recall 的【最小整数】
        # 我们不预设上限，而是看数据到底需要多少
        for bias in range(0, 100):
            y_safe = y_final_base + bias
            y_safe = np.clip(y_safe, val_a, val_b)

            cov = np.mean(y_safe >= y_true)

            # 记录当前状态
            # print(f"    - 尝试 Bias={bias}, 覆盖率={cov*100:.2f}%") # 调试用，可注释
            #不应该是target_recall,应该是1
            if cov >= target_recall:
                best_bias = bias
                final_cov = cov
                break

        # ---------------------------------------------------------
        # 最终配置报告
        # ---------------------------------------------------------
        print(f"\n{'=' * 80}")
        print(f"最终优化报告 (OPTIMIZATION REPORT)")
        print(f"{'=' * 80}")
        print(f"1. 斜率参数 (k)      : {best_k:.5f} (刻画数据形状)")
        print(f"2. 安全系数 (Bias)   : +{best_bias} (刻画数据位置)")
        print(f"3. 最终覆盖率        : {final_cov * 100:.2f}% (目标: {target_recall * 100}%)")
        print(f"{'=' * 80}")

        # ---------------------------------------------------------
        # 深度对比表
        # ---------------------------------------------------------
        print("\n>>> 深度抽样验证 (Double Check)")
        # QID | 真实需 | 基础值 | 最终值 | 安全扩增比 | 真实偏差比 | 状态
        header = f"{'QID':<6} | {'真实需':<4} | {'基础值':<4} | {'最终值':<4} | {'安全扩增比':<9} | {'真实偏差比':<9} | {'状态'}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # 智能抽样：简单、中等、困难、极难 各选几个
        idxs_easy = np.where(y_true == 100)[0]
        idxs_mid = np.where((y_true > 100) & (y_true <= 110))[0]
        idxs_hard = np.where(y_true > 110)[0]

        # 防止数量不够报错
        selection = []
        if len(idxs_easy) > 0: selection.extend(np.random.choice(idxs_easy, min(3, len(idxs_easy)), replace=False))
        if len(idxs_mid) > 0: selection.extend(np.random.choice(idxs_mid, min(3, len(idxs_mid)), replace=False))
        if len(idxs_hard) > 0: selection.extend(np.random.choice(idxs_hard, min(4, len(idxs_hard)), replace=False))

        if not selection: selection = np.random.choice(len(y_true), 10)

        for idx in selection:
            t_val = int(y_true[idx])
            p_base = int(y_final_base[idx])
            p_final = int(np.clip(p_base + best_bias, val_a, val_b))

            status = "✅ 达标" if p_final >= t_val else "❌ 漏搜"

            # 1. 安全扩增比 (Final / Base)
            if p_base > 0:
                ratio_safety = p_final / p_base
            else:
                ratio_safety = 1.0

            # 2. 真实偏差比 (Final / True)
            if t_val > 0:
                ratio_true = p_final / t_val
            else:
                ratio_true = 1.0

            print(
                f"{idx:<6} | {t_val:<6} | {p_base:<6} | {p_final:<6} | {ratio_safety:<10.2f}x | {ratio_true:<10.2f}x | {status}")

        print("-" * len(header))
        print("指标解读:")
        print("1. [安全扩增比]: 算法为了安全，在模型预测基础上额外加了多少。")
        print("2. [真实偏差比]: 最终结果相比真实需求浪费了多少。")
        print(f"   * 如果Bias较大，说明模型本身方差大，必须保留这个余量才能保证不漏搜。")


if __name__ == "__main__":
    main()