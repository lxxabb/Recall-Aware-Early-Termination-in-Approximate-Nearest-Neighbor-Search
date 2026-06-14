
#同一个qid和r的行数不能超过10行，r=1的行数不能超过20行

import pandas as pd
import numpy as np

in_file = "/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main-data/data/et_training_data/early-stop-training/GLOVE100/k100/M16_efC500_efS500_qs10000_li2_visitedPoints.txt"
out_file = "/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main-data/data/et_training_data/early-stop-training/GLOVE100/k100/M16_efC500_efS500_qs10000_li2_visitedPoints_filtered_group_cap.txt"

df = pd.read_csv(in_file)

# 保留原顺序
df["_orig_order"] = range(len(df))

# 确保 r 为浮点
df["r"] = pd.to_numeric(df["r"], errors="coerce")

# 按原顺序排序
df = df.sort_values("_orig_order")

# 每个 (qid, r) 组合的行号计数，从 0 开始
df["cnt"] = df.groupby(["qid", "r"]).cumcount()

# 条件：
# r == 1 → 允许 0~19 共20条
# r != 1 → 允许 0~9 共10条
mask = np.where(np.isclose(df["r"], 1.0), df["cnt"] < 20, df["cnt"] < 10)

result = df[mask].drop(columns=["_orig_order", "cnt"])

# 保持原顺序输出
result = result.sort_index()

result.to_csv(out_file, index=False)

print(f"完成！输出 {len(result)} 行")
