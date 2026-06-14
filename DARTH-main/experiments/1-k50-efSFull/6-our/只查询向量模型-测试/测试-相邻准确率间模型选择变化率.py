import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义准确率列表和对应的 Recall 文件后缀（根据你提供的路径规律）
tr_list = [95, 96, 97, 98, 99]
# 注意：95 和 96 的 Recall 文件后缀不同，这里假设 97+ 是延续 96 的命名规则或需要你补充
# 请根据你的实际文件名修改下面的 suffix_map
suffix_map = {
    95: "_60.txt",  # tr0.95 的特殊后缀
    96: "_64.txt",  # tr0.96 的特殊后缀
    # 假设 97, 98, 99 没有特殊后缀，或者请替换为实际后缀
    97: "_68.txt",
    98: "_71.txt",
    99: "_67.txt"
}

# 存储每个 tr 下的耗时数据
times_recall = {}
times_cns = {}

base_path_recall = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k100/efS500_qs10000_tr0."
base_path_cns = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/SIFT1M/k100/efS500_qs10000_tr0."

for tr in tr_list:
    recall_path = f"{base_path_recall}{tr}{suffix_map[tr]}"
    cns_path = f"{base_path_cns}{tr}.txt" # 通常 CNS 文件没有特殊后缀

    df_r = pd.read_csv(recall_path)
    df_c = pd.read_csv(cns_path)

    times_recall[tr] = df_r["elaps_ms"].values  # 转为 numpy 数组便于计算
    times_cns[tr] = df_c["elaps_ms"].values

# ---------------------------------------------------------
# 分析 1：统计每个准确率下，选择 CNS 作为最优模型的比例
# ---------------------------------------------------------
optimal_cns_ratio = {}
change_rates = {} # 存储相邻层级的变化率

print("=== 各准确率下最优模型分布 ===")
for tr in tr_list:
    # 0: Recall 最优, 1: CNS 最优
    choice = (times_cns[tr] < times_recall[tr]).astype(int)
    ratio = choice.mean() # CNS 被选中的比例
    optimal_cns_ratio[tr] = ratio
    print(f"tr{tr}: CNS 最优比例 = {ratio:.3f} ({int(ratio*100)}%)")

# ---------------------------------------------------------
# 分析 2：比较相邻准确率之间的选择变化 (e.g., 95->96, 96->97...)
# ---------------------------------------------------------
print("\n=== 相邻准确率间模型选择变化率 ===")
for i in range(len(tr_list)-1):
    tr_curr = tr_list[i]
    tr_next = tr_list[i+1]

    choice_curr = (times_cns[tr_curr] < times_recall[tr_curr]).astype(int)
    choice_next = (times_cns[tr_next] < times_recall[tr_next]).astype(int)

    # 计算发生变化的比例
    change_rate = (choice_curr != choice_next).mean()
    change_rates[f"{tr_curr}_{tr_next}"] = change_rate
    print(f"tr{tr_curr} -> tr{tr_next}: 模型选择变化率 = {change_rate:.3f} ({int(change_rate*100)}%)")