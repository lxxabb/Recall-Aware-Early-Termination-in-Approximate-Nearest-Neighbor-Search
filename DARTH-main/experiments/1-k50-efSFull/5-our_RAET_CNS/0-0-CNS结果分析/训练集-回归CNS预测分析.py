import pandas as pd
import numpy as np
# 保存当前设置（可选）
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# ===============================
# 1. 文件路径（按你实际路径修改）
# ===============================
pred_file = "/root/DARTH-main-data/result/raet-cns-regression_CNSFull/SIFT1M/k100/train_data/efS200_qs10000_tr0.99.txt"
label_file = "/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efs500_min_ef_per_query_s10000_0.99.csv"

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


# ===============================
# NEW 0. 分桶放大因子学习
# ===============================
# ===============================
# NEW 0. 等宽分桶放大因子学习
# ===============================
def learn_equal_width_bucket_scaler(
        df,
        bucket_width=10,
        max_ef=200,
        min_samples_per_bucket=30,
        max_scale=2.0
):
    """
    等宽分桶： [0,10), [10,20), ...
    """
    df = df.copy()
    df = df[(df["y_pred"] > 0) & (df["y_true"] > 0)]

    # 计算桶编号
    df["bucket_id"] = (df["y_pred"] // bucket_width).astype(int)

    bucket_rows = []

    max_bucket = int(max_ef // bucket_width)

    for b in range(max_bucket + 1):
        l = b * bucket_width
        r = (b + 1) * bucket_width

        g = df[(df["y_pred"] >= l) & (df["y_pred"] < r)]
        if len(g) < min_samples_per_bucket:
            continue

        mean_pred = g["y_pred"].mean()
        mean_true = g["y_true"].mean()

        scale = mean_true / mean_pred

        # 安全约束
        scale = max(scale, 1.0)
        scale = min(scale, max_scale)

        bucket_rows.append({
            "bucket_id": b,
            "pred_min": l,
            "pred_max": r,
            "scale": scale,
            "count": len(g),
            "mean_pred": mean_pred,
            "mean_true": mean_true
        })

    bucket_df = pd.DataFrame(bucket_rows)
    return bucket_df


# ===============================
# NEW 1. 应用分桶放大
# ===============================
# ===============================
# NEW 1. 应用等宽分桶放大
# ===============================
def apply_equal_width_scaler(
        y_pred,
        bucket_df,
        bucket_width=10,
        max_ef=200
):
    b = int(y_pred // bucket_width)

    row = bucket_df[bucket_df["bucket_id"] == b]
    if len(row) == 0:
        return max_ef

    scale = row.iloc[0]["scale"]
    return min(max_ef, int(y_pred * scale))


# ===============================
# NEW 2. 学习分桶放大因子
# ===============================
bucket_df = learn_equal_width_bucket_scaler(
    df,
    bucket_width=10,   # 强烈建议 10 或 20
    max_ef=MAX_EF,
    min_samples_per_bucket=30
)

print("\n=== Bucket scaling table ===")
print(bucket_df)

# ===============================
# NEW 3. 生成放大后的预测值
# ===============================
df["y_calibrated"] = df["y_pred"].apply(
    lambda x: apply_equal_width_scaler(x, bucket_df, MAX_EF)
)

df["error"] = df["y_calibrated"] - df["y_true"]
df["under"] = df["error"] < 0

under_df = df[df["under"]].copy()
under_df["under_gap"] = under_df["y_true"] - under_df["y_calibrated"]

print("\n====对预测值进行放大后的结果====")
print("\n=============================")
# ===============================
# 4. 核心指标 ①：低估比例
# ===============================
under_ratio = df["under"].mean()
print("\n=== Under-estimation ratio ===")
print(f"Under-estimated queries: {under_ratio:.4%}")

if len(under_df) == 0:
    print("No under-estimation detected 🎉")
else:
    print(under_df["under_gap"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    for th in [1, 5, 10, 20, 30]:
        ratio = (under_df["under_gap"] >= th).mean()
        print(f"P(under_gap >= {th}) = {ratio:.4%}")


