import pandas as pd

# ====== 参数 ======
input_file = "/root/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k100/efS1000_qs10000.txt"
output_file = "/root/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k100/efS1000_qs10000_shrink.txt"

# r == 1 时最多 10 条，否则最多 5 条
MAX_NORMAL = 5
MAX_R1 = 10

# ====== 读取数据 ======
df = pd.read_csv(input_file)

# 保证 r 是 float（防止字符串问题）
df["r"] = df["r"].astype(float)

# ====== 组内编号 ======
# 对每个 (qid, r) 生成 0,1,2,...
df["_cnt"] = df.groupby(["qid", "r"]).cumcount()

# ====== 按规则过滤 ======
mask = (
        ((df["r"] == 1.0) & (df["_cnt"] < MAX_R1)) |
        ((df["r"] != 1.0) & (df["_cnt"] < MAX_NORMAL))
)

df_new = df[mask].drop(columns="_cnt")

# ====== 保存 ======
df_new.to_csv(output_file, index=False)

print(f"原始行数: {len(df)}")
print(f"删减后行数: {len(df_new)}")
print("完成 ✅")
