import pandas as pd
# 1w 训练集  标签分布 等于100的占比，
#SIFT1M
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.99.csv"  #39.35%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.98.csv"  #52.44%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.97.csv"  #62.30%
#
# #GLOVE100
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_0.99.csv"#65.53%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_0.98.csv"#77.85%
#
# #GIST 放弃二分类，CNS=k占比太小，二分类模型作用有限
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.99.csv"#6.84%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.98.csv"#9.63%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.97.csv"#12.77%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.96.csv"#15.81%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.95.csv"#18.53%
#
# #DEEP
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.99.csv"#37.63%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.98.csv"#47.52%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.97.csv"#56.16%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.96.csv"#62.71%
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.95.csv"#68.12%
#
# 10w 训练集  标签分布 等于100的占比，
#SIFT1M
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_s100000_0.99.csv"  #39.35%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_s100000_0.98.csv"  #52.44%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_s100000_0.97.csv"  #62.30%

#GLOVE100
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_s100000_0.99.csv"#65.53%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_s100000_0.98.csv"#77.85%

#GIST 放弃二分类，CNS=k占比太小，二分类模型作用有限
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_s100000_0.99.csv"#6.84%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_s100000_0.98.csv"#9.63%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_s100000_0.97.csv"#12.77%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_s100000_0.96.csv"#15.81%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_s100000_0.95.csv"#18.53%

#DEEP
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_s100000_0.99.csv"#37.63%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_s100000_0.98.csv"#47.52%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_s100000_0.97.csv"#56.16%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_s100000_0.96.csv"#62.71%
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_s100000_0.95.csv"#68.12%

min_cns_df = pd.read_csv(label_path)

# 计算 min_efSearch = 100 的占比
total = len(min_cns_df)
greater_than_100 = (min_cns_df['min_efSearch'] == 100).sum()
ratio = greater_than_100 / total

print(f"总样本数: {total}")
print(f"min_efSearch == 100 的数量: {greater_than_100}")
print(f"占比: {ratio:.4f} ({ratio * 100:.2f}%)")