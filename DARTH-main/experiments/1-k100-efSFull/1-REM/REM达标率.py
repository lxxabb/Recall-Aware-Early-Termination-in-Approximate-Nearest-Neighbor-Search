import pandas as pd

file_path = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/REM/T2I1M/k100/efS1000_qs10000_tr0.95.txt"  # 改成你的路径

# 读取数据
df = pd.read_csv(file_path)

# 取召回率列
recall = df["r"]

# 阈值列表
thresholds = [0.95, 0.96, 0.97, 0.98, 0.99]

print("总查询数:", len(recall))
print()

# 统计达标率
for t in thresholds:
    hit_rate = (recall >= t).mean()
    print(f"召回率 ≥ {t:.2f} 的比例: {hit_rate:.4f}")