import pandas as pd

label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.99.csv"  #(60.65%)
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.98.csv"  #(47.56%)
label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/min_ef_per_query_0.97.csv"  #(37.70%)


# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_0.99.csv"#(34.47%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/min_ef_per_query_0.98.csv"#(22.15%)
#
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.99.csv"#(93.16%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.98.csv"#(90.37%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.97.csv"#(87.23%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.96.csv"#(84.19%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/min_ef_per_query_0.95.csv"#(81.47%)
#
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.99.csv"#(62.37%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.98.csv"#(52.48%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.97.csv"#(43.84%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.96.csv"#(37.29%)
# label_path = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/min_ef_per_query_0.95.csv"#(31.88%)










min_cns_df = pd.read_csv(label_path)

# 计算 min_efSearch > 100 的占比
total = len(min_cns_df)
greater_than_100 = (min_cns_df['min_efSearch'] > 100).sum()
ratio = greater_than_100 / total

print(f"总样本数: {total}")
print(f"min_efSearch > 100 的数量: {greater_than_100}")
print(f"占比: {ratio:.4f} ({ratio * 100:.2f}%)")