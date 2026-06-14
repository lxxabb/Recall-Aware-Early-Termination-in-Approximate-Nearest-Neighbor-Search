import pandas as pd
import os
import re
# 显示所有行
pd.set_option('display.max_rows', None)
# 显示所有列
pd.set_option('display.max_columns', None)
# 不限制列宽（避免列内容被截断）
pd.set_option('display.width', None)
# 不限制单元格内容长度
pd.set_option('display.max_colwidth', None)
# ========================
# 所有 CSV 文件路径
# ========================
file_list = [
    # -------- SIFT1M --------
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.95.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.97.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.98.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.99.csv",

    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.90.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.92.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.94.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.98.csv",

    # -------- DEEP10M --------
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/efs750_min_ef_per_query_s10000_0.95.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/efs750_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/efs750_min_ef_per_query_s10000_0.97.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/efs750_min_ef_per_query_s10000_0.98.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/efs750_min_ef_per_query_s10000_0.99.csv",

    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k50/efs750_min_ef_per_query_s10000_0.90.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k50/efs750_min_ef_per_query_s10000_0.92.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k50/efs750_min_ef_per_query_s10000_0.94.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k50/efs750_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k50/efs750_min_ef_per_query_s10000_0.98.csv",
    # -------- gist --------
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/efs1000_min_ef_per_query_s10000_0.95.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/efs1000_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/efs1000_min_ef_per_query_s10000_0.97.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/efs1000_min_ef_per_query_s10000_0.98.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/efs1000_min_ef_per_query_s10000_0.99.csv",

    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k50/efs1000_min_ef_per_query_s10000_0.90.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k50/efs1000_min_ef_per_query_s10000_0.92.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k50/efs1000_min_ef_per_query_s10000_0.94.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k50/efs1000_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k50/efs1000_min_ef_per_query_s10000_0.98.csv",
    # -------- glove --------
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/efs500_min_ef_per_query_s10000_0.95.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/efs500_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/efs500_min_ef_per_query_s10000_0.97.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/efs500_min_ef_per_query_s10000_0.98.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/efs500_min_ef_per_query_s10000_0.99.csv",

    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k50/efs500_min_ef_per_query_s10000_0.90.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k50/efs500_min_ef_per_query_s10000_0.92.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k50/efs500_min_ef_per_query_s10000_0.94.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k50/efs500_min_ef_per_query_s10000_0.96.csv",
    "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k50/efs500_min_ef_per_query_s10000_0.98.csv",
]

# ========================
# 正则：解析 dataset / k / recall
# ========================
#1代表预测准确率方法好，0代表预测CNS方法好
labels = [
    #sift
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    #deep
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    #gist
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    #glove
    1, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
]

# 正则解析 dataset, k, recall
dataset_re = re.compile(r"training/([^/]+)/k\d+")
k_re       = re.compile(r"/k(\d+)/")
recall_re  = re.compile(r"_([0-9]+\.[0-9]+)\.csv$")

rows = []

for i, file_path in enumerate(file_list):
    df = pd.read_csv(file_path)
    cv = df["min_efSearch"].std(ddof=1) / df["min_efSearch"].mean()

    dataset = dataset_re.search(file_path).group(1)
    k_val   = k_re.search(file_path).group(1)
    recall  = recall_re.search(file_path).group(1)

    label = labels[i]  # 按顺序对应

    rows.append({
        "dataset": dataset,
        "k": int(k_val),
        "recall": float(recall),
        "cv": cv,
        "label": label
    })

df_all = pd.DataFrame(rows)
print(df_all)

