import pandas as pd

# ========== 参数 ==========
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.95.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.96.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.97.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.98.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.99.txt"

# filename = "/root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.95.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.96.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.97.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.98.txt"
# filename = "/root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.99.txt"

#!/usr/bin/env python3
import argparse
import pandas as pd

EPS = 1e-6  # 防止浮点误差


def main():
    parser = argparse.ArgumentParser(
        description="Statistics of stability at first reaching target recall"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to stability log file"
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        required=True,
        help="Target recall threshold, e.g. 0.95 or 1.0"
    )

    args = parser.parse_args()

    filename = args.file
    TARGET_RECALL = args.target_recall

    # 1. 读取数据
    df = pd.read_csv(
        filename,
        header=None,
        names=["query", "step", "stability", "recall"]
    )

    # 2. 只保留 recall >= TARGET_RECALL 的行
    hit = df[df["recall"] >= TARGET_RECALL - EPS]

    if hit.empty:
        print(f"[WARN] No record reaches recall >= {TARGET_RECALL}")
        return

    # 3. 按 query + step 排序，保证“第一次”
    hit = hit.sort_values(["query", "step"])

    # 4. 每个 query 取第一次达到 target recall 的那一行
    first_hit = hit.groupby("query", as_index=False).first()

    # 5. stability
    stability = first_hit["stability"]

    # 6. 输出统计
    print(f"===== Stability statistics =====")
    print(f"file          : {filename}")
    print(f"target_recall : {TARGET_RECALL}")
    print(f"mean          : {stability.mean():.3f}")
    print(f"median        : {stability.median():.3f}")

if __name__ == "__main__":
    main()
