import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import dask.dataframe as dd
from sklearn.metrics import r2_score

# ================= 路径配置 (自动适配你的环境) =================
BASE_DIR = r"D:\Learnnnn\dachuang\DARTH-main-data"
ANN_DIR = r"D:\Learnnnn\dachuang\ann-data"

# 你想验证哪个数据集？
DATASET_NAME = "GLOVE100" 
# 验证集还是测试集？(训练用的是 qs10000，这里我们也先用 qs10000 验证一下训练效果，或者换成测试集)
# 注意：你的文件夹结构里可能叫 'test' 或者跟训练集在同一个目录下
# 这里假设我们先验证训练集本身（或者你有单独的测试文件）
DATA_MODE = "test"  # 或者 "validation"
QUERY_NUM = 10000   # 你的文件名里是 10000

# 向量文件路径
QUERY_VEC_FILE = f"{ANN_DIR}/GLOVE100/query.10K.fvecs" 
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def predict_scheme1(model_path, param_path, X):
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在: {model_path}")
        return None
    if not os.path.exists(param_path):
        print(f"[Error] 参数文件不存在: {param_path}")
        return None

    # 1. 加载参数
    with open(param_path, 'r') as f:
        params = json.load(f)
    
    k_scale = params['final_k']
    min_val = params['min_val']
    max_val = params['max_val']

    # 2. 加载 Booster
    bst = lgb.Booster(model_file=model_path)

    # 3. 获取原始分值 z (raw_score=True)
    z = bst.predict(X, raw_score=True)

    # 4. 应用 Scheme 1 公式
    y_pred = min_val + (max_val - min_val) * sigmoid(k_scale * z)

    return y_pred, params

def compute_metrics(y_true, y_pred, k_min, cns_max):
    y_pred_clipped = np.clip(y_pred, k_min, cns_max)
    
    mse = mean_squared_error(y_true, y_pred_clipped)
    mae = mean_absolute_error(y_true, y_pred_clipped)
    
    # 核心指标：覆盖率 (预测值 >= 真实值 的比例)
    # 【新增】计算 R2 Score
    r2 = r2_score(y_true, y_pred_clipped)
    safe_mask = y_pred_clipped >= y_true
    coverage = np.mean(safe_mask)
    
    # 欠拟合的平均误差
    under_mask = y_pred_clipped < y_true
    if np.sum(under_mask) > 0:
        under_estimation_mae = np.mean(y_true[under_mask] - y_pred_clipped[under_mask])
    else:
        under_estimation_mae = 0.0

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,    # 添加到返回字典
        "Coverage": coverage,
        "Under_MAE": under_estimation_mae
    }

def read_fvecs(file_path, limit=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"向量文件未找到: {file_path}")
        
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        dim = first_entry[0]
        f.seek(0, 2)
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)
        if limit is not None and limit < num_vectors:
            num_vectors = limit
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)
        vectors = data[:, 1:]
    return vectors, int(dim)

def main():
    k_val = 100
    efS = 500
    d = 100
    n_estimators = 100
    
    # 构造特征列名 (必须与训练时完全一致)
    qid_feat=["qid"]
    index_metric_feats = ["step", "dists", "inserts"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy"]
    cosine_feats = ["cosine_mean","cosine_variance","cosine_direction_entropy"]
    all_feats_cols = qid_feat + index_metric_feats + neighbor_distances_feats + neighbor_stats_feats + cosine_feats
    
    columns_to_load = ["elaps_ms"] + all_feats_cols + ["r", "feats_collect_time_ms"]
    
    target_recalls = [0.98, 0.99]
    
    print(f"=== 开始验证 Scheme 1 模型: {DATASET_NAME} ===")

    # 1. 读取数据文件
    # 假��你的测试文件路径结构与训练类似，或者这里直接用训练数据做演示验证
    # 如果你有单独的 test 文件夹，请修改这里
    data_path = f"D:/Learnnnn/dachuang/data/glove数据集/测试集/efS500_qs10000.txt"
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"[Error] 数据文件未找到: {data_path}")
        return

    ddf = dd.read_csv(data_path, usecols=columns_to_load)
    ddf = ddf[ddf["dists"] % 1 == 0]
    data_all = ddf.compute()
    
    # 2. 读取 Query 向量
    print(f"Loading query vectors from: {QUERY_VEC_FILE}")
    query_dims, _ = read_fvecs(QUERY_VEC_FILE, limit=QUERY_NUM)
    
    dims_cols = [f"d{i+1}" for i in range(d)]
    per_query_dimensions_df = pd.DataFrame(query_dims, index=data_all["qid"].unique()[:len(query_dims)], columns=dims_cols)
    
    # 3. 循环验证
    for target_recall in target_recalls:
        print(f"\n--- Target Recall: {target_recall} ---")
        
        # 加载标签
        label_path = f"D:/Learnnnn/dachuang/data/glove数据集/测试集/min_ef_per_query_{target_recall}.csv"
        
        if not os.path.exists(label_path):
            print(f"Skipping: Label file not found: {label_path}")
            continue
            
        labels_df = pd.read_csv(label_path).set_index("qid")
        
        # 合并数据
        merged_X = data_all.merge(per_query_dimensions_df, left_on="qid", right_index=True)
        merged_all = merged_X.merge(labels_df, left_on="qid", right_index=True, how="inner")
        
        feature_cols = ["step", "dists", "inserts", "first_nn_dist", "nn_dist", "furthest_dist",
                        "avg_dist", "variance", "avg_hop", "density", "entry_query_dist_ratio",
                        "dist_distribution_entropy", "skewness", "kurtosis", "energy",
                        "cosine_mean", "cosine_variance", "cosine_direction_entropy"] + dims_cols
        
        X_test = merged_all[feature_cols].values
        y_true = merged_all["min_efSearch"].values
        
        # 加载刚才训练好的模型
        model_file = f"{BASE_DIR}/predictor_models/raet-cns-regression/{DATASET_NAME}/k{k_val}/efS{efS}_s{QUERY_NUM}_nestim{n_estimators}_all_feats_{target_recall}_Scheme1.txt"
        param_file = model_file.replace(".txt", "_params.json")
        
        result = predict_scheme1(model_file, param_file, X_test)
        
        if result:
            y_pred, params = result
            metrics = compute_metrics(y_true, y_pred, params['min_val'], params['max_val'])
            
            print(f"Learned k: {params['final_k']:.4f}")
            print(f"Metrics:")
            print(f"  Coverage (Safe Ratio): {metrics['Coverage']:.2%}  <-- 重点关注这个！")
            print(f"  R2 Score:              {metrics['R2']:.4f}")  # 打印它
            print(f"  Under-estimation MAE:  {metrics['Under_MAE']:.2f}")
            print(f"  MSE: {metrics['MSE']:.2f}")

            # 打印几个样本看看
            print("\n  Sample (True vs Pred):")
            for t, p in zip(y_true[:5], y_pred[:5]):
                print(f"  {t:3d}  |  {p:.1f}")

if __name__ == "__main__":
    main()