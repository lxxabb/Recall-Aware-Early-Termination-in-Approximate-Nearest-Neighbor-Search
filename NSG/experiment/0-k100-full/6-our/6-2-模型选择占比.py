import pandas as pd

# 1. 读取你的数据文件（替换成你的文件路径，比如 data.csv）
# step:0 是选择了召回率方法，step：1 是选择了CNS预测方法
#sift
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/SIFT1M/k50/efS500_qs10000_tr0.90.txt")
df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/SIFT1M/k100/efS500_s10000_tr0.95.txt")
#glove
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GLOVE100/k50/efS500_qs10000_tr0.90.txt")
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GLOVE100/k100/efS500_s10000_tr0.95.txt")
#deep
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/DEEP10M/k100/efS750_s10000_tr0.95.txt")
#gist
# df = pd.read_csv("/home/extra_home/lxx23125236/ali/NSG-data/results/our/GIST1M/k100/efS1000_s1000_tr0.95.txt")
# 2. 统计 selector 列中 0 和 1 的数量,0是召回率，1是CNS
selector_counts = df["selector"].value_counts()

# 3. 计算占比（百分比）
selector_ratio = df["selector"].value_counts(normalize=True) * 100

# 4. 输出结果
print("===== selector 取值统计 =====")
for value in sorted(selector_counts.index):
    count = selector_counts[value]
    ratio = selector_ratio[value]
    print(f"selector = {value}: 数量 {count}，占比 {ratio:.2f}%")