import sys
import json
import pandas as pd
import numpy as np
import os

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <input_csv> <output_json> <target_recall>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
target_recall=float(sys.argv[3])
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File not found: {input_file}")

df = pd.read_csv(input_file)

# -------- 数据清洗 --------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.fillna(0, inplace=True)

results = {}

# -------- 模型调用统计 --------
mean_predictor_calls = df["r_predictor_calls"].mean()

results["predictor_calls"] = {
    "average_calls": float(mean_predictor_calls),
    "describe": df["r_predictor_calls"].describe().to_dict()
}

# -------- 只统计 用到模型 的行 --------
df_used = df[df["r_predictor_calls"] != 0]
total_predict_time = df_used["r_predictor_time_ms"].sum()
total_predict_calls = df_used["r_predictor_calls"].sum()

avg_time_per_call = (
    total_predict_time / total_predict_calls
    if total_predict_calls != 0 else 0
)

results["predictor_time"] = {
    "total_predict_time_ms": float(total_predict_time),
    "total_predict_calls": int(total_predict_calls),
    "avg_time_per_call_ms": float(avg_time_per_call)
}

# -------- Early Stop --------
total_rows = len(df)
early_stop_ratio = (df["r_predicted"] > target_recall).sum() / total_rows

results["early_stop"] = {
    "total_rows": int(total_rows),
    "early_stop_ratio": float(early_stop_ratio)
}

# -------- r_actual --------
below_target_recall = (df["r_actual"] < target_recall).sum() / total_rows

results["r_actual_stats"] = {
    "ratio_r_actual_below_target_recall": float(below_target_recall)
}

# -------- 预测误差 --------
df_pred = df[df["r_predicted"] != 0].copy()

if len(df_pred) == 0:
    results["prediction_error"] = {
        "message": "No rows where r_predicted != 0"
    }
else:
    errors = df_pred["r_predicted"] - df_pred["r_actual"]

    mae = errors.abs().mean()
    mse = (errors ** 2).mean()
    rmse = np.sqrt(mse)
    bias = errors.mean()

    valid = df_pred["r_actual"] != 0
    mape = (errors[valid].abs() / df_pred["r_actual"][valid]).mean() * 100

    ss_res = (errors ** 2).sum()
    ss_tot = ((df_pred["r_actual"] - df_pred["r_actual"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    corr = df_pred[["r_predicted", "r_actual"]].corr().iloc[0, 1]

    results["prediction_error"] = {
        "num_samples": int(len(df_pred)),
        "bias": float(bias),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "r2": float(r2),
        "pearson_corr": float(corr)
    }

# -------- RDE / TDR / NRS 统计 --------
metrics = ["RDE", "TDR", "NRS"]
results["metrics"] = {}

for m in metrics:
    col = df[m].copy()

    # 统计 nan 占比
    nan_ratio = col.isna().mean()

    # 只对非 nan 统计
    col_valid = col.dropna()

    metric_stats = {
        "num_valid": int(len(col_valid)),
        "nan_ratio": float(nan_ratio),
        "describe": col_valid.describe().to_dict()
    }

    # NRS 里往往用 -1 表示 not found，这个可以额外算
    if m == "NRS":
        neg_one_ratio = (col_valid == -1).mean()
        metric_stats["neg_one_ratio"] = float(neg_one_ratio)

        # 如果你不想把 -1 算进去：
        col_valid = col_valid[col_valid != -1]
        metric_stats["describe_excluding_-1"] = col_valid.describe().to_dict()

    results["metrics"][m] = metric_stats

# -------- 保存 JSON --------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved analytics to {os.path.abspath(output_file)}")
