import pandas as pd

# 1. 读取你的数据文件（替换成你的文件路径，比如 data.csv）
# step:0 是选择了召回率方法，step：1 是选择了CNS预测方法
#sift
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k50/efS500_qs10000_tr0.90.txt")
#glove
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/GLOVE100/k50/efS500_qs10000_tr0.90.txt")
#deep
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/DEEP10M/k50/efS750_qs10000_tr0.90.txt")
#gist
df = pd.read_csv("/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/GIST1M/k50/efS1000_qs1000_tr0.90.txt")

# 2. 统计 step 列中 0 和 1 的数量
step_counts = df["step"].value_counts()

# 3. 计算占比（百分比）
step_ratio = df["step"].value_counts(normalize=True) * 100

# 4. 输出结果
print("===== step 取值统计 =====")
for value in sorted(step_counts.index):
    count = step_counts[value]
    ratio = step_ratio[value]
    print(f"step = {value}: 数量 {count}，占比 {ratio:.2f}%")