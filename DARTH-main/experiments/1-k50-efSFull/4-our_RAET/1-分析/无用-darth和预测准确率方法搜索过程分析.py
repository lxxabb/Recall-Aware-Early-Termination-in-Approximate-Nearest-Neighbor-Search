# darth-sift 95结果路径：/root/DARTH-main-data/result/darth/SIFT1M/k100/efS500_qs10000_tr0.95_ipi1174_mpi235.txt
#预测准确率95 搜索路径： /root/DARTH-main-data/result/raet/SIFT1M/k100/efS500_qs10000_tr0.95_59.txt
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# 文件路径
darth_file = "/root/DARTH-main-data/result/darth/SIFT1M/k100/efS500_qs10000_tr0.95_ipi1174_mpi235.txt"
raet_file  = "/root/DARTH-main-data/result/raet/SIFT1M/k100/efS500_qs10000_tr0.95_59.txt"
# raet_file  = "/root/DARTH-main-data/result/raet/SIFT1M/k100/times/efS500_qs10000_tr0.95_59_t1_test.txt"
# 读取 CSV（注意你的文件是逗号分隔）
df_darth = pd.read_csv(darth_file)
df_raet  = pd.read_csv(raet_file)

# 只保留我们关心的列（避免同名列冲突）
df_darth = df_darth[["qid", "r_predicted"]].rename(
    columns={"r_predicted": "r_predicted_darth"}
)
df_raet = df_raet[["qid", "r_predicted"]].rename(
    columns={"r_predicted": "r_predicted_raet"}
)

# 按 qid 合并
df = pd.merge(df_darth, df_raet, on="qid", how="inner")

# 条件筛选：
# darth 已经满足 0.95，但 raet 还没满足
mask = (df["r_predicted_darth"] >= 0.95) & (df["r_predicted_raet"] < 0.95)
early_stop_diff = df[mask]

# 结果统计
print("==================首先找到，不同方法darth早停单raet未早停的数据点有哪些===========================")
print(f"满足条件的数据点数量: {len(early_stop_diff)}")

# 看前几个
print(early_stop_diff.head(10))

# # 保存下来，方便你后续画图或做 case study
# early_stop_diff.to_csv(
#     "darth_early_stop_but_raet_not.csv",
#     index=False
# )
print("==================开始对每个数据点的搜索过程详细说明==========================")
#darth搜索过程路径： /root/DARTH-main/experiments/1-k100-efSFull/5-our_RAET_CNS/0-0-CNS结果分析/无用-sift_darth.txt
# 其中内容为： csv_file << query_idx <<"," << ndis <<"," << query_predictor_calls <<"," << predictedRecallOut <<"\n";
#raet 搜索路径为/root/DARTH-main/experiments/1-k100-efSFull/5-our_RAET_CNS/0-0-CNS结果分析/无用-sift_raet.txt
#其中内容为： csv_file << query_idx <<"," << ndis <<","<< stability_counts <<","<<outer_layer_predict_recall<<","<< pre_now_stability_insert
#<<"," <<prev_recall<<","<< current_insert_count <<"," << query_predictor_calls <<"," << predictedRecallOut <<"\n";

#
# darth_file = "/root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/1-分析/无用-sift_darth.txt"
# raet_file  = "/root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/1-分析/无用-sift_raet.txt"
# raet_file_new  = "/root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/1-分析/sift_raet_new.txt"
# df_darth = pd.read_csv(darth_file)
# df_raet  = pd.read_csv(raet_file)
# df_raet_new  = pd.read_csv(raet_file_new)
# print("==================darth==========================")
# print(df_darth)
# print("==================raet==========================")
# print(df_raet)
# print("==================raet_new==========================")
# print(df_raet_new)
# print(0)