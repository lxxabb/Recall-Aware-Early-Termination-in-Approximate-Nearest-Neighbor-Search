#特征路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efS500_qs10000.txt
#recall标签路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/train/SIFT1M/k100/efS500_qs10000_tr0.95_60.txt
#CNS标签路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/train/SIFT1M/k100/efS500_qs10000_tr0.95.txt
import os

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time
import json
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
# fea_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efS500_qs10000.txt"
fea_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k100/train-data/efS500_qs10000_tr0.95.txt"
recall_label_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/train/SIFT1M/k100/efS500_qs10000_tr0.95_60.txt"
cns_label_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/train/SIFT1M/k100/efS500_qs10000_tr0.95.txt"

fea_df = pd.read_csv(fea_file)
recall_label_df = pd.read_csv(recall_label_file)
cns_label_df = pd.read_csv(cns_label_file)
input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                   "avg_dist", "variance","avg_hop","entry_query_dist_ratio","dist_distribution_entropy",
                   "cosine_mean","cosine_variance","cosine_direction_entropy"]
X_train = fea_df[input_features].values
# 读取两个方法的时间
recall_time = recall_label_df["elaps_ms"].values
cns_time = cns_label_df["elaps_ms"].values

# 检查长度是否一致
assert len(recall_time) == len(cns_time), "两个标签文件长度不一致"

# 生成二分类标签
# cns更快 -> 0
# recall更快 -> 1
y_all = np.where(cns_time <= recall_time, 0, 1)

num_total = len(y_all)
num_0 = np.sum(y_all == 0)
num_1 = np.sum(y_all == 1)

print(f"Total samples: {num_total}")
print(f"Label 0 (CNS faster): {num_0} ({num_0 / num_total:.4f})")
print(f"Label 1 (Recall faster): {num_1} ({num_1 / num_total:.4f})")

y_train = y_all
SEED = 42
n_estimators = 100
model_conf = {
    "raet-cns": lgb.LGBMRegressor(objective='binary', random_state=SEED, n_estimators=n_estimators, verbose = -1),
}
model = model_conf["raet-cns"]
model_train_time_start = time.time()
model.fit(X_train, y_train)
model_train_time = time.time() - model_train_time_start

model_params = model.get_params()
learning_rate = model_params["learning_rate"]

# 生成特征重要性 DataFrame
feature_importances = pd.DataFrame({
    'Feature': input_features,
    'Importance': model.feature_importances_
})

# 计算占比（百分比）
total_importance = feature_importances['Importance'].sum()
feature_importances['Importance_Ratio'] = feature_importances['Importance'] / total_importance
# 排序
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 构造保存路径（和模型同名更清晰）
fi_file = f"/home/extra_home/lxx23125236/ali/DARTH-main-data/feature_importance/our/SIFT1M/k100/efS500_s10000_nestim100_all_feats_recall_selector_0.95.csv"
# 保存到文件
os.makedirs(os.path.dirname(fi_file), exist_ok=True)
feature_importances.to_csv(fi_file, index=False)
print(f"特征重要性保存到: {fi_file}")

model_file = f"/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/our/SIFT1M/k100/efS500_s10000_nestim100_all_feats_recall_selector_0.95.txt"
os.makedirs(os.path.dirname(model_file), exist_ok=True)
model.booster_.save_model(model_file)

print(f"模型保存到: {model_file}")
print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")
