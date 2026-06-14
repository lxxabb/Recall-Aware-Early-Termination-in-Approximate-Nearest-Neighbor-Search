import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

def read_fvecs(file_path, limit=None):
    """读取 fvecs 格式的向量文件"""
    with open(file_path, 'rb') as f:
        # 读取维度
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        # 计算向量数量
        vector_size = (dim + 1) * 4
        file_size = os.fstat(f.fileno()).st_size
        num_vectors = file_size // vector_size

        # 应用限制
        if limit is not None:
            num_vectors = min(num_vectors, limit)

        # 读取数据
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)
        vectors = data[:, 1:] # 去掉第一列的维度信息
        return vectors, dim

# ================== 配置参数 ==================
ds_name = "SIFT1M"
d = 128
k = 100
efS = 500
tr = 0.98
query_num = 10000

data_dir = "/home/extra_home/lxx23125236/ali/DARTH-main-data"

# --- 路径配置 ---
# 1. 模型路径
model_input_dir = f"{data_dir}/predictor_models/our/query_vec_only/{ds_name}/k{k}"
model_file = f"{model_input_dir}/efS{efS}_qs10000_0.98_query_vec_model.txt"

# 2. 测试查询向量路径
query_vec_path = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_query.fvecs"

# 3. 真实标签路径 (测试集)
recall_label_file = f"{data_dir}/result/raet/{ds_name}/k{k}/efS{efS}_qs{query_num}_tr{tr:.2f}.txt"
cns_label_file = f"{data_dir}/result/raet-cns-regression_CNSFull/{ds_name}/k{k}/efS{efS}_qs{query_num}_tr{tr:.2f}.txt"

# ================== 1. 加载模型 ==================
print(f"Loading model from: {model_file}")
model = lgb.Booster(model_file=model_file)

# ================== 2. 读取测试数据并预测 ==================
print("Loading and predicting on test queries...")
X_test, test_dim = read_fvecs(query_vec_path, limit=query_num)
assert test_dim == d, f"Dim mismatch"

# 预测概率
y_pred_prob = model.predict(X_test)
# 转换为类别: 0 (CNS更快), 1 (Recall更快)
# 阈值 0.5: 如果预测概率 > 0.5，认为 Recall 更快
pred = (y_pred_prob > 0.5).astype(int)

# ================== 3. 读取真实标签并计算准确率 ==================
print("Calculating accuracy...")
recall_df = pd.read_csv(recall_label_file)
cns_df = pd.read_csv(cns_label_file)

T_recall = recall_df["elaps_ms"].values
T_cns = cns_df["elaps_ms"].values

# --- 统一标签定义 ---
# 0: CNS 更快 (T_cns <= T_recall)
# 1: Recall 更快 (T_recall < T_cns)
best = np.where(T_cns <= T_recall, 0, 1)

# --- 准确率计算 ---
correct = (pred == best).sum()
accuracy = correct / len(pred)
print(f"Selector Accuracy: {accuracy:.4f} ({correct}/{len(pred)})")

# ================== 3.5 生成模型选择后的新文件 ==================
print("Generating selected query file...")

# 基本检查
if len(recall_df) != len(cns_df) or len(recall_df) != len(pred):
    raise ValueError("recall_df / cns_df / pred 长度不一致。")

if not recall_df["qid"].equals(cns_df["qid"]):
    raise ValueError("recall_label_file 和 cns_label_file 的 qid 顺序不一致，不能直接按行选择。")

selected_df = pd.DataFrame({
    "qid": recall_df["qid"],
    "elaps_ms": np.where(pred == 1, recall_df["elaps_ms"], cns_df["elaps_ms"]),
    "r_actual": np.where(pred == 1, recall_df["r_actual"], cns_df["r"])
})

selected_output_file = f"{data_dir}/result/our/{ds_name}/k{k}/efS{efS}_qs{query_num}_tr{tr:.2f}_selected_queries.txt"
selected_df.to_csv(selected_output_file, index=False)

print(f"Selected query file saved to: {selected_output_file}")
print(selected_df.head())


# ================== 4. 性能耗时分析 ==================
def total_seconds(avg_ms, q_num):
    return avg_ms * q_num / 1000

# Selector Ideal: 根据预测结果选择的时间
# 如果 pred=0 (选CNS) -> T_cns; pred=1 (选Recall) -> T_recall
selector_time_ideal = np.where(pred == 0, T_cns, T_recall)

# Selector Optimal: 理论上每次都能选到最快的
optimal_time = np.minimum(T_recall, T_cns)

total_queries = len(pred)

print("\n--- Performance Comparison (Total Time in Seconds) ---")
print(f"Selector Ideal total time (s): {total_seconds(selector_time_ideal.mean(), total_queries):.2f}")
print(f"Selector Optimal total time (s): {total_seconds(optimal_time.mean(), total_queries):.2f}")
print(f"Always Recall total time (s): {total_seconds(T_recall.mean(), total_queries):.2f}")
print(f"Always CNS total time (s): {total_seconds(T_cns.mean(), total_queries):.2f}")