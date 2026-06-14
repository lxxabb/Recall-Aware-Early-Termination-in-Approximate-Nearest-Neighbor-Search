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
        vector_size = (dim + 1) * 4  # 每个向量包含一个维度头
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
d = 128          # 向量维度
s = 10000        # 查询数量 (前10000个)
k = 100
efS = 500
tr = 0.95

# 数据路径配置
data_dir = "/home/extra_home/lxx23125236/ali/DARTH-main-data"

# 查询向量文件路径 (根据数据集选择)
if ds_name == "SIFT1M":
    query_vec_path = "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_learn.fvecs"
elif ds_name == "GIST1M":
    query_vec_path = "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_learn.fvecs"
elif ds_name == "GLOVE100":
    query_vec_path = "/home/extra_home/lxx23125236/ann-data/GLOVE100/learn.100K.fvecs"
elif ds_name == "DEEP10M":
    query_vec_path = "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs"
else:
    raise ValueError(f"Unsupported dataset: {ds_name}")

# 训练数据集文件路径
recall_label_file = f"{data_dir}/result/raet/train/{ds_name}/k{k}/efS{efS}_qs{s}_tr{tr:.2f}_60.txt"
cns_label_file = f"{data_dir}/result/raet-cns-regression_CNSFull/train/{ds_name}/k{k}/efS{efS}_qs{s}_tr{tr:.2f}.txt"

# 输出路径
model_output_dir = f"{data_dir}/predictor_models/our/query_vec_only/{ds_name}/k{k}"
fi_output_dir = f"{data_dir}/feature_importance/our/query_vec_only/{ds_name}/k{k}"
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(fi_output_dir, exist_ok=True)

# ================== 1. 直接读取查询向量 (前 s 个) ==================
print(f"Reading first {s} query vectors from {query_vec_path}...")
X_train, dim_read = read_fvecs(query_vec_path, limit=s)

# 验证形状
assert X_train.shape == (s, d), f"Shape mismatch: expected ({s}, {d}), got {X_train.shape}"
print(f"Query vectors loaded. Shape: {X_train.shape}")

# ================== 2. 读取标签并生成二分类标签 ==================
print("Loading labels...")
recall_label_df = pd.read_csv(recall_label_file)
cns_label_df = pd.read_csv(cns_label_file)

recall_time = recall_label_df["elaps_ms"].values
cns_time = cns_label_df["elaps_ms"].values

assert len(recall_time) == len(cns_time) == s, "Label length mismatch"

# 生成二分类标签: 0 (CNS更快), 1 (Recall更快)
y_train = np.where(cns_time <= recall_time, 0, 1)

num_total = len(y_train)
num_0 = np.sum(y_train == 0)
num_1 = np.sum(y_train == 1)

print(f"Total samples: {num_total}")
print(f"Label 0 (CNS faster): {num_0} ({num_0/num_total:.4f})")
print(f"Label 1 (Recall faster): {num_1} ({num_1/num_total:.4f})")

# ================== 3. 训练模型 ==================
print("Training LightGBM model...")
SEED = 42
n_estimators = 100

model = lgb.LGBMRegressor(
    objective='binary',
    random_state=SEED,
    n_estimators=n_estimators,
    verbose=-1
)

model_train_time_start = time.time()
model.fit(X_train, y_train)
model_train_time = time.time() - model_train_time_start

# ================== 4. 保存特征重要性 ==================
feature_names = [f"vec_dim_{i}" for i in range(d)]
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
})
importance_df['Importance_Ratio'] = importance_df['Importance'] / importance_df['Importance'].sum()
importance_df = importance_df.sort_values(by='Importance', ascending=False)

fi_file = f"{fi_output_dir}/efS{efS}_qs{s}_0.95_query_vec_importance.csv"
importance_df.to_csv(fi_file, index=False)
print(f"Feature importance saved to: {fi_file}")

# ================== 5. 保存模型 ==================
model_file = f"{model_output_dir}/efS{efS}_qs{s}_0.95_query_vec_model.txt"
model.booster_.save_model(model_file)
print(f"Model saved to: {model_file}")

print(f"Training completed. Time taken: {model_train_time:.2f}s")