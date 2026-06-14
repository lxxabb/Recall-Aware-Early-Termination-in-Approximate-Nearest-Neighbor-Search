import os
import pandas as pd

# -----------------------------
# 配置
# -----------------------------

base_dir = "/home/extra_home/lxx23125236/ali/NSG-data/results"

methods = [
    # "laet",
    # "darth",
    # "raet",
    "raet_CNS_regression"
]

datasets = {
    "SIFT1M": {
        "efs": 500,
        "s": 10000
    },
    # "DEEP10M": {
    #     "efs": 750,
    #     "s": 10000
    # },
    # "GIST1M": {
    #     "efs": 1000,
    #     "s": 1000
    # },
    "GLOVE100": {
        "efs": 500,
        "s": 10000
    },
    # "T2I1M": {
    #     "efs": 1000,
    #     "s": 10000
    # }
}

targets = ["0.95", "0.96", "0.97", "0.98", "0.99"]

# -----------------------------
# 结果保存
# -----------------------------

results = []

# -----------------------------
# 开始统计
# -----------------------------

for method in methods:

    for dataset, params in datasets.items():

        efs = params["efs"]
        s = params["s"]

        for target in targets:

            filename = f"k100/efS{efs}_s{s}_tr{target}.txt"
            filepath = os.path.join(
                base_dir,
                method,
                dataset,
                filename
            )

            # 判断文件是否存在
            if not os.path.exists(filepath):
                print(f"文件不存在: {filepath}")
                continue

            try:
                # 读取文件
                df = pd.read_csv(filepath)

                # 实际达到目标召回率
                hit_count = (df["r_actual"] >= float(target)).sum()

                total_count = len(df)

                hit_ratio = f"{hit_count / total_count * 100:.2f}%"

                results.append({
                    "method": method,
                    "dataset": dataset,
                    "target_recall": target,
                    "total_queries": total_count,
                    "hit_queries": hit_count,
                    "hit_ratio": hit_ratio
                })

                print(
                    f"{method} | {dataset} | {target} "
                    f"-> {hit_ratio}"
                )

            except Exception as e:
                print(f"读取失败: {filepath}")
                print(e)

# -----------------------------
# 保存结果
# -----------------------------

result_df = pd.DataFrame(results)

output_file = "/home/extra_home/lxx23125236/ali/NSG/experiment/0-k100-full/recall_statistics.csv"

result_df.to_csv(output_file, index=False)

print("\n统计完成")
print(result_df)

print(f"\n结果已保存到: {output_file}")