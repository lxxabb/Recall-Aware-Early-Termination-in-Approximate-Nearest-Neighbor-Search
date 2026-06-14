import pandas as pd
import numpy as np

# ===============================
# 1. 文件路径（按你实际路径修改）
# ===============================
pred_file = "/root/DARTH-main-data/result/raet-cns-regression_CNSFull/SIFT1M/k100/efS200_qs10000_tr0.99.txt"
label_file = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/test/efs500_min_ef_per_query_s10000_0.99.csv"

MAX_EF = 200

# ===============================
# 2. 读取数据
# ===============================
# 预测结果（只用 CNS_predicted + qid）
pred_df = pd.read_csv(pred_file, usecols=["qid", "CNS_predicted"])
pred_df.rename(columns={"CNS_predicted": "y_pred"}, inplace=True)

# 真实标签
label_df = pd.read_csv(label_file)
label_df = label_df.set_index("qid")

# 截断真实标签
label_df["y_true"] = label_df["min_efSearch"].clip(upper=MAX_EF)
hard_query_ratio=(label_df["y_true"] == 200).mean()
print("困难查询点占比",hard_query_ratio)
eazy_query_ratio=(label_df["y_true"] == 100).mean()
print("简单查询点占比",eazy_query_ratio)
# 合并
df = pred_df.merge(
    label_df[["y_true", "min_efSearch"]],
    left_on="qid",
    right_index=True,
    how="inner"
)

print("Total queries:", len(df))

# ===============================
# 3. 误差分析
# ===============================
df["error"] = df["y_pred"] - df["y_true"]
df["under"] = df["error"] < 0

# ===============================
# 4. 核心指标 ①：低估比例
# ===============================
under_ratio = df["under"].mean()
print("\n=== Under-estimation ratio ===")
print(f"Under-estimated queries: {under_ratio:.4%}")

# ===============================
# 5. 核心指标 ②：低估幅度分布
# ===============================
under_df = df[df["under"]].copy()
under_df["under_gap"] = under_df["y_true"] - under_df["y_pred"]

print("\n=== Under-estimation gap statistics ===")
if len(under_df) == 0:
    print("No under-estimation detected 🎉")
else:
    print(under_df["under_gap"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    for th in [1, 5, 10, 20, 30]:
        ratio = (under_df["under_gap"] >= th).mean()
        print(f"P(under_gap >= {th}) = {ratio:.4%}")

# ===============================
# 6. 核心指标 ③：hard query（y_true = MAX_EF）
# ===============================
hard_df = df[df["y_true"] == MAX_EF]

print("\n=== Hard queries (y_true == {}) ===".format(MAX_EF))
print("Hard query count:", len(hard_df))

if len(hard_df) > 0:
    hard_under_ratio = (hard_df["y_pred"] < MAX_EF).mean()
    print(f"P(y_pred < {MAX_EF} | y_true = {MAX_EF}) = {hard_under_ratio:.4%}")

# ===============================
# 7. 理论可行性检查（是否 ef 上限不够）
# ===============================
true_over_max_ratio = (df["min_efSearch"] > MAX_EF).mean()

print("\n=== Feasibility check ===")
print(f"P(min_efSearch > {MAX_EF}) = {true_over_max_ratio:.4%}")

if true_over_max_ratio > 0.01:
    print("⚠️ efSearch upper bound may be insufficient for 0.99 recall")
else:
    print("Upper bound likely sufficient")