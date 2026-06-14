import pandas as pd

# #darth glove 0.95  0.7461
# # 文件路径
file_path = "/root/DARTH-main-data/result/darth/GLOVE100/k100/efS500_qs10000_tr0.95_ipi439_mpi88.txt"
# 读取 CSV 文件
df = pd.read_csv(file_path)

# 计算 r_actual >= 0.95 的行数
count_high_r_actual = (df['r_actual'] >= 0.95).sum()

# 总行数
total_count = len(df)

# 占比
ratio = count_high_r_actual / total_count
# 计算平均准确率
average_r_actual = df['r_actual'].mean()

print(f"总行数: {total_count}")
print(f"r_actual >= 0.95 的行数: {count_high_r_actual}")
print(f"占比: {ratio:.4f}")  # 输出四位小数
print(f"平均准确率 (r_actual)：{average_r_actual:.4f}")
min_r_actual = df['r_actual'].min()
print(f"r_actual 的最低值：{min_r_actual:.6f}")
#
#
# #raet glove 0.95  0.7461
# # 文件路径
# file_path = "/root/DARTH-main-data/result/raet-cns-regression_CNSFull/GLOVE100/k100/efS500_qs10000_tr0.95.txt"
#
# # 读取 CSV 文件
# df = pd.read_csv(file_path)
#
# # 计算 r_actual >= 0.95 的行数  0.9536
# count_high_r_actual = (df['r'] >= 0.95).sum()
#
# # 总行数
# total_count = len(df)
#
# # 占比
# ratio = count_high_r_actual / total_count
# # 计算平均准确率
# average_r_actual = df['r'].mean()
#
# print(f"总行数: {total_count}")
# print(f"r_actual >= 0.95 的行数: {count_high_r_actual}")
# print(f"占比: {ratio:.4f}")  # 输出四位小数
# print(f"平均准确率 (r_actual)：{average_r_actual:.4f}")
# min_r_actual = df['r'].min()
# print(f"r_actual 的最低值：{min_r_actual:.6f}")

#adaptnn  0.6582
# file_path = "/root/DARTH-main-data/result/laet/GLOVE100/k100/efS500_qs10000_tr0.95_f200.txt"
# # 读取 CSV 文件
# df = pd.read_csv(file_path)
#
# # 计算 r_actual >= 0.95 的行数
# count_high_r_actual = (df['r'] >= 0.95).sum()
#
# # 总行数
# total_count = len(df)
#
# # 占比
# ratio = count_high_r_actual / total_count
# # 计算平均准确率
# average_r_actual = df['r'].mean()
#
# print(f"总行数: {total_count}")
# print(f"r_actual >= 0.95 的行数: {count_high_r_actual}")
# print(f"占比: {ratio:.4f}")  # 输出四位小数
# print(f"平均准确率 (r_actual)：{average_r_actual:.4f}")
# min_r_actual = df['r'].min()
# print(f"r_actual 的最低值：{min_r_actual:.6f}")

# #rem  0.6582
# file_path = "/root/DARTH-main-data/result/REM/GLOVE100/k100/efS500_qs10000_tr0.95.txt"
# # 读取 CSV 文件
# df = pd.read_csv(file_path)
#
# # 计算 r_actual >= 0.95 的行数
# count_high_r_actual = (df['r'] >= 0.95).sum()
#
# # 总行数
# total_count = len(df)
#
# # 占比
# ratio = count_high_r_actual / total_count
# # 计算平均准确率
# average_r_actual = df['r'].mean()
#
# print(f"总行数: {total_count}")
# print(f"r_actual >= 0.95 的行数: {count_high_r_actual}")
# print(f"占比: {ratio:.4f}")  # 输出四位小数
# print(f"平均准确率 (r_actual)：{average_r_actual:.4f}")
# min_r_actual = df['r'].min()
# print(f"r_actual 的最低值：{min_r_actual:.6f}")



