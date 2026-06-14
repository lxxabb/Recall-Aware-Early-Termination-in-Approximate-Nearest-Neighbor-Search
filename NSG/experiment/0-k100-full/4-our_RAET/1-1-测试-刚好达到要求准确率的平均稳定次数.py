import pandas as pd

EPS = 1e-6  # 防止浮点误差


# =========================
# 🔧 手动设置参数（直接改）
# =========================
FILENAME = "/root/NSG-data/results/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.99.txt"
TARGET_RECALL = 0.99


def main():
    print("[DEBUG] filename =", FILENAME)
    print("[DEBUG] target_recall =", TARGET_RECALL)

    # 1️⃣ 读取数据
    df = pd.read_csv(FILENAME)

    # 👉 打断点/检查
    # print(df.head())

    # 2️⃣ 只保留 recall >= TARGET_RECALL 的行
    hit = df[df["r"] >= TARGET_RECALL - EPS]

    if hit.empty:
        print(f"[WARN] No record reaches recall >= {TARGET_RECALL}")
        return

    # 3️⃣ 按 query + step 排序，保证“第一次”
    hit = hit.sort_values(["qid", "step"])

    # 4️⃣ 每个 query 取第一次达到 target recall 的那一行
    first_hit = hit.groupby("qid", as_index=False).first()

    # 👉 debug 用
    # print(first_hit.head())

    # 5️⃣ stability
    stability = first_hit["stability"]

    # 6️⃣ 输出统计
    print("===== Stability statistics =====")
    print(f"file          : {FILENAME}")
    print(f"target_recall : {TARGET_RECALL}")
    print(f"count         : {len(stability)}")
    print(f"mean          : {stability.mean():.3f}")
    print(f"median        : {stability.median():.3f}")
    print(f"min           : {stability.min():.3f}")
    print(f"max           : {stability.max():.3f}")


if __name__ == "__main__":
    main()
