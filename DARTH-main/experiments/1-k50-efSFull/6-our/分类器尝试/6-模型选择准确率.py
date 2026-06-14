#预测准确率每个数据点时间路径在/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k100/efS500_qs10000_tr0.95_60.txt
#预测CNS每个数据点时间路径在/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/SIFT1M/k100/efS500_qs10000_tr0.95.txt
#通过模型选择算法得到的每个数据点时间路径在/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k100/efS500_qs10000_tr0.95.txt
#比较的是列是elaps_ms,如果选择器中的 r_predicted 列的值为-1那么是预测CNS，否则是预测准确率
import numpy as np
import pandas as pd

recall_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k100/efS500_qs10000_tr0.95_60.txt"
cns_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/SIFT1M/k100/efS500_qs10000_tr0.95.txt"
selector_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k100/efS500_qs10000_tr0.95.txt"
# selector_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k100/times/efS500_qs10000_tr0.95_t1.txt"
recall_df = pd.read_csv(recall_file)
cns_df = pd.read_csv(cns_file)
selector_df = pd.read_csv(selector_file)

T_recall = recall_df["elaps_ms"]
T_cns = cns_df["elaps_ms"]
T_selector = selector_df["elaps_ms"]

r_pred = selector_df["r_predicted"]

# selector预测
pred = (r_pred == 0.0).astype(int)   # 1=CNS 0=Recall

# 真实最优
best = (T_cns < T_recall).astype(int)

# 计算准确率
correct = (pred == best).sum()
accuracy = correct / len(pred)

print("Selector accuracy:", accuracy)

optimal_time = np.minimum(T_recall, T_cns)
selector_time_ideal = np.where(pred == 1, T_cns, T_recall)

query_num = len(selector_df)
def total_seconds(avg_ms):
    #单位由ms变为s，要除以1000
    return avg_ms * query_num / 1000
#从模型选择器算法执行中获取的每个数据点花费时间
print("Selector total time (s):", total_seconds(T_selector.mean()))
#理想情况的意思是，根据选择器结果，从预测CNS和预测recall中取时间，得到最终总时间。不包括模型选择器二分类模型的预测时间和加载模型时间
print("Selector ideal total time (s):", total_seconds(selector_time_ideal.mean()))
#最优情况，意思是，每个数据点取预测CNS和预测recall的最小时间
print("Selector optimal total time (s):", total_seconds(optimal_time.mean()))
print("Always Recall total time (s):", total_seconds(T_recall.mean()))
print("Always CNS total time (s):", total_seconds(T_cns.mean()))

