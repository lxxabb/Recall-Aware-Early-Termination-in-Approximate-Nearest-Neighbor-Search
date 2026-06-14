import pandas as pd

# 读取你的CSV文件（把文件名改成你真实的文件）
#sift
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k50/efS500_qs10000_tr0.90.txt")
#glove
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/GLOVE100/k50/efS500_qs10000_tr0.90.txt")
#deep
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/DEEP10M/k50/efS750_qs10000_tr0.90.txt")
#gist
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/GIST1M/k50/efS1000_qs1000_tr0.90.txt")

# 统计 r_actual > 0.90 的行数
count_above = (df["r_actual"] > 0.90).sum()

# 总行数
total_count = len(df)

# 计算占比（百分比）
ratio = (count_above / total_count) * 100

# 输出结果
print(f"r_actual > 0.90 的行数：{count_above}")
print(f"总行数：{total_count}")
print(f"占比：{ratio:.2f}%")