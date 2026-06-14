import pandas as pd
import numpy as np
# 读入文件，r_predictor_time_ms模型平均预测时间0.35273ms，1万个查询点总时间为秒，换算下来，时间为3527.3ms,,1s=1000ms,即3s,模型预测时间为3s
# df = pd.read_csv("/root/DARTH-main-data/data/result/early-stop-testing/SIFT1M/k100/times/M32_efC500_efS128_qs10000_tr0.98_ipi1569_mpi313_t1.txt")
# df = pd.read_csv("/root/DARTH-main-data/data/result/early-stop-testing/SIFT1M/k100/times/M32_efC500_efS128_qs10000_tr0.98_stability48_t1.txt")
# df = pd.read_csv("/root/DARTH-main-data/data/result/early-stop-testing/SIFT1M/k100/times/M32_efC500_efS128_qs10000_tr0.98_stability48_t1_删除查询点向量特征.txt")
# df = pd.read_csv("/root/DARTH-main-data/result/delete_query.test")
# df = pd.read_csv("/root/DARTH-main-data/result/delete_insertK_entrypoint.test")
# df = pd.read_csv("/root/DARTH-main-data/result/delete_entrpoint.test")
# df = pd.read_csv("/root/DARTH-main-data/result/delete_insertK.test")
df = pd.read_csv("/root/DARTH-main-data/result/delete_nothing.test")

# 将 inf / -inf 转成 NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 再把 NaN 填成 0
df.fillna(0, inplace=True)
# # 计算 模型预测时间 的平均值
# mean_predictor_time = df["r_predictor_time_ms"].mean()

# 计算 r_predictor_calls的平均值。模型调用次数
mean_predictor_calls = df["r_predictor_calls"].mean()
print(f"\n=== 模型 统计 ===")
# print("Average r_predictor_time_ms模型平均预测时间:", mean_predictor_time)
print("Average predictor_calls模型平均被调用次数:", mean_predictor_calls)
print(df["r_predictor_time_ms"].describe())
print(df["r_predictor_calls"].describe())

# 只统计 r_predictor_calls != 0 的行
mask = df["r_predictor_calls"] != 0
df_used = df[mask]
# 平均到 “每一次调用” 上（推荐）
total_predict_time = df_used["r_predictor_time_ms"].sum()
total_predict_calls = df_used["r_predictor_calls"].sum()
avg_time_per_call = total_predict_time / total_predict_calls if total_predict_calls != 0 else 0
print(f"模型总预测时间(ms): {total_predict_time}")
print(f"模型总调用次数: {total_predict_calls}")
print(f"平均到单次调用的预测时间(ms): {avg_time_per_call}")

num_used_rows = len(df_used)

# === 统计 r_predicted 早停率 ===
total_rows = len(df)

early_stop_mask = df["r_predicted"] > 0.98
early_stop_count = early_stop_mask.sum()

early_stop_ratio = early_stop_count / total_rows if total_rows != 0 else 0

print("\n=== Early Stop 统计 ===")
print(f"总行数: {total_rows}")
print(f"r_predicted > 0.98 的行数: {early_stop_count}")
print(f"早停率: {early_stop_ratio:.4%} ({early_stop_ratio:.4f})")
# === our Early Stop 统计 ===
# 总行数: 10000
# r_predicted > 0.98 的行数: 6423
# 早停率: 64.2300% (0.6423)

# === 新增：统计 r_actual < 0.98 的占比 ===
total_rows = len(df)
below_098 = df["r_actual"] < 0.98
count_below_098 = below_098.sum()
ratio_below_098 = count_below_098 / total_rows
print(f"\n=== r_actual 统计 ===")
print(f"总行数: {total_rows}")
print(f"r_actual < 0.98 的行数: {count_below_098}")
print(f"r_actual < 0.98 的占比: {ratio_below_098:.4%} ({ratio_below_098:.4f})") #efS500,sift1M k100,不达标的占比17.27

# === 只统计 r_predicted != 0 的行，计算预测误差 ===
df_pred = df[df["r_predicted"] != 0].copy()
n_pred = len(df_pred)

if n_pred == 0:
    print("\n没有 r_predicted != 0 的行，无法计算误差指标")
else:
    # 误差：预测 - 实际
    errors = df_pred["r_predicted"] - df_pred["r_actual"]

    mae = errors.abs().mean()                          # 平均绝对误差
    mse = (errors ** 2).mean()                        # 均方误差
    rmse = np.sqrt(mse)                               # RMSE
    bias = errors.mean()                              # 偏差（是否系统性高估/低估）

    # MAPE（避免除0问题）
    valid = df_pred["r_actual"] != 0
    mape = (errors[valid].abs() / df_pred["r_actual"][valid]).mean() * 100

    # R^2
    ss_res = (errors ** 2).sum()
    ss_tot = ((df_pred["r_actual"] - df_pred["r_actual"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # 相关系数（衡量线性一致性）
    corr = df_pred[["r_predicted", "r_actual"]].corr().iloc[0,1]

    print("\n=== 模型预测误差统计（仅 r_predicted != 0）===")
    print(f"样本数: {n_pred}")
    print(f"Bias(平均误差): {bias:.6f}")
    print(f"MAE(平均绝对误差): {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE(%): {mape:.4f}")
    print(f"R²: {r2:.6f}")
    print(f"Pearson相关系数: {corr:.6f}")