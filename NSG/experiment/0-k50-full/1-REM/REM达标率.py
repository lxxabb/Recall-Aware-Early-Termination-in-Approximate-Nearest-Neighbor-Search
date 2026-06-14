import pandas as pd

# 文件路径
file_path = "/home/extra_home/lxx23125236/ali/NSG-data/results/REM/T2I1M/k50/efS1000_s10000_tr0.90.txt"

# 读取（不信任表头，手动指定列）
df = pd.read_csv(file_path, header=0)

# 如果列数是4，说明中间多了一列（你这个就是）
if df.shape[1] == 4:
    df.columns = ["qid", "dummy", "r_actual", "search_time"]
elif df.shape[1] == 3:
    df.columns = ["qid", "r_actual", "search_time"]
else:
    raise ValueError("列数异常，请检查文件格式")

# 提取召回率
recall = df["r_actual"]

# 阈值
thresholds = [0.90, 0.92, 0.94, 0.96, 0.98]

print("总查询数:", len(recall))

for t in thresholds:
    hit_rate = (recall >= t).mean()
    print(f"召回率 ≥ {t:.2f} 的比例: {hit_rate:.4f}")