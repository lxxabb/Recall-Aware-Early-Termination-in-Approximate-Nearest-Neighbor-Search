import pandas as pd

file = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efs500_min_ef_per_query_s10000_0.96.csv"
df = pd.read_csv(file)

total = len(df)
count_50 = (df["min_efSearch"] == 50).sum()
count_500 = (df["min_efSearch"] == 500).sum()

print("total:", total)
print("50 count:", count_50, "ratio:", count_50 / total)
print("500 count:", count_500, "ratio:", count_500 / total)

# HNSW图索引
# sift 90 k=50 0.7788
# sift 92 k=50 0.705
# sift 94 k=50 0.6185
# sift 96 k=50 0.6185
# sift 98 k=50 0.5046

file = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.99.csv"
df = pd.read_csv(file)

total = len(df)
count_100 = (df["min_efSearch"] == 100).sum()
count_500 = (df["min_efSearch"] == 500).sum()

print("total:", total)
print("100 count:", count_100, "ratio:", count_100 / total)
print("500 count:", count_500, "ratio:", count_500 / total)


# sift 95 k=100 0.7754
# sift 96 k=100 0.7106
# sift 97 k=100 0.623
# sift 98 k=100 0.5244
# sift 99 k=100 0.3935