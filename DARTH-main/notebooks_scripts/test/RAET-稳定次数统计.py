import pandas as pd

# 你的日志文件
log_file = "/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main-data/统计文件/stability_counts.csv"

df = pd.read_csv(log_file, header=None)
df.columns = ["query_idx", "step", "stability_counts", "recall"]

# 排序，保证时序一致
df = df.sort_values(["query_idx", "step"])

results = []

for q, g in df.groupby("query_idx"):
    run = 0
    for v in g["stability_counts"]:
        if v != 0:
            run += 1
        else:
            if run > 0:
                results.append({"query_idx": q, "run_length": run})
            run = 0
    # 结尾如果正好停在非0段
    if run > 0:
        results.append({"query_idx": q, "run_length": run})

runs_df = pd.DataFrame(results)

print(runs_df.head())
print("总共统计到连续非0 段：", len(runs_df))