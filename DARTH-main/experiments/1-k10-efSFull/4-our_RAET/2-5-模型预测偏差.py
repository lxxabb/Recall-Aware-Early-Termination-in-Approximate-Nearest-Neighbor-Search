import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# ======================
# 1. 路径配置
# ======================
data_file = "/root/DARTH-main-data/data/et_training_data/early-stop-training/SIFT1M/k100/efS500_qs10000.txt"

model_file = "/root/DARTH-main-data/predictor_models/raet/SIFT1M/k100/efS500_s10000_nestim100_all_feats_Noquery.txt"
# ↑ 注意替换成你真实存在的模型文件名

# ======================
# 2. 读取数据
# ======================
data_df = pd.read_csv(data_file)

# 你使用的特征（和训练时保持一致）
feature_cols = [
    "step",
    "dists",
    "inserts",
    "visited_points",
    "nn_dist",
    "furthest_dist",
    "avg_dist",
    "variance"
]

X_all = data_df[feature_cols].values.astype(np.float64)
y_all = data_df["r"].values.astype(np.float64)

# ======================
# 3. 只筛选 r == 0.95 的样本
# ======================
target_r = 0.99
mask = np.isclose(y_all, target_r)

X_target = X_all[mask]
y_target = y_all[mask]

print(f"Total samples with r = {target_r}: {len(y_target)}")

if len(y_target) == 0:
    raise ValueError("没有找到 r=0.95 的样本，请检查数据")

# ======================
# 4. 加载 LightGBM 模型
# ======================
booster = lgb.Booster(model_file=model_file)

# ======================
# 5. 模型预测
# ======================
y_pred = booster.predict(X_target)

# ======================
# 6. 误差统计
# ======================
abs_error = np.abs(y_pred - y_target)

max_error = abs_error.max()
mean_error = abs_error.mean()
std_error = abs_error.std()

print("====== Prediction Error Statistics (r = 0.95) ======")
print(f"Max absolute error : {max_error:.6f}")
print(f"Mean absolute error: {mean_error:.6f}")
print(f"Std absolute error : {std_error:.6f}")

# ======================
# 7. （可选）更详细分析
# ======================
print("\nSome example predictions:")
for i in range(min(5, len(y_pred))):
    print(f"pred={y_pred[i]:.6f}, true={y_target[i]:.6f}, abs_err={abs_error[i]:.6f}")
#0.95 0.032
#0.96 0.0304
#0.97 0.028
#0.98 0.027
#0.99 0.024
