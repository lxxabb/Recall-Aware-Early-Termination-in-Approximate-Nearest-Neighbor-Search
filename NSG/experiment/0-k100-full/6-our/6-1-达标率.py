import pandas as pd

# 读取你的CSV文件（把文件名改成你真实的文件）
#sift
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/SIFT1M/k50/efS500_qs10000_tr0.90.txt")
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/SIFT1M/k100/efS500_s10000_tr0.95.txt")
#glove
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GLOVE100/k50/efS500_qs10000_tr0.90.txt")
df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GLOVE100/k100/efS500_s10000_tr0.95.txt")

#deep
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/DEEP10M/k100/efS750_s10000_tr0.95.txt")
#gist
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GIST1M/k100/efS1000_s1000_tr0.95.txt")


# 统计 r_actual > 0.95 的行数
count_above = (df["r_actual"] > 0.95).sum()

# 总行数
total_count = len(df)

# 计算占比（百分比）
ratio = (count_above / total_count) * 100

# 输出结果
print(f"r_actual > 0.95 的行数：{count_above}")
print(f"总行数：{total_count}")
print(f"占比：{ratio:.2f}%")