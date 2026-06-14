import os
import pandas as pd

# -----------------------------
# 配置
# -----------------------------

base_dir = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result"

methods = [
    # "REM",
    "darth",
    "laet",
    "raet",
    "raet-cns-regression_CNSFull"
]

datasets = {
    # "SIFT1M": {"efs": 500, "qs": 10000},
    # "DEEP10M": {"efs": 750, "qs": 10000},
    # "GIST1M": {"efs": 1000, "qs": 1000},
    # "GLOVE100": {"efs": 500, "qs": 10000},
    "T2I1M": {"efs": 1000, "qs": 10000}
}

# ⚠️ 你现在要统计这些阈值
targets = [0.90, 0.92, 0.94, 0.96, 0.98]

results = []

# -----------------------------
# 开始统计
# -----------------------------

for method in methods:
    for dataset, params in datasets.items():

        efs = params["efs"]
        qs = params["qs"]

        for target in targets:

            filename = f"efS{efs}_qs{qs}_tr{target:.2f}.txt"

            filepath = os.path.join(
                base_dir,
                method,
                dataset,
                "k50",
                filename
            )

            if not os.path.exists(filepath):
                print(f"❌ 文件不存在: {filepath}")
                continue

            try:
                df = pd.read_csv(filepath)

                # 🔥 自动兼容 r 或 r_actual
                if "r" in df.columns:
                    recall = df["r"]
                elif "r_actual" in df.columns:
                    recall = df["r_actual"]
                else:
                    raise ValueError("找不到召回率列")

                total = len(recall)
                hit = (recall >= target).sum()
                ratio = hit / total

                results.append({
                    "method": method,
                    "dataset": dataset,
                    "target": target,
                    "total": total,
                    "hit": hit,
                    "hit_ratio": ratio
                })

                print(f"{method} | {dataset} | {target:.2f} -> {ratio:.4f}")

            except Exception as e:
                print(f"❌ 读取失败: {filepath}")
                print(e)

# -----------------------------
# 保存结果
# -----------------------------

result_df = pd.DataFrame(results)

output_path = "/home/extra_home/lxx23125236/ali/DARTH-main/experiments/1-k50-efSFull/recall_k50_statistics.csv"
result_df.to_csv(output_path, index=False)

print("\n✅ 统计完成")
print(result_df)
print(f"\n📁 已保存到: {output_path}")