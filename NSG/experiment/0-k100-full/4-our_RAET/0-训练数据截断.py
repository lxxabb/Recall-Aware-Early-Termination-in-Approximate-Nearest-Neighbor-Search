import pandas as pd

# 1. 读取文件
# input_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/SIFT1M/k100/efS500_qs10000.txt"
# output_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/SIFT1M/k100/efS500_qs10000_r_ge_0.90.txt"
# 
# input_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/GLOVE100/k100/efS500_qs10000.txt"
# output_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/GLOVE100/k100/efS500_qs10000_r_ge_0.90.txt"

# input_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/DEEP10M/k100/efS750_qs10000.txt"
# output_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/DEEP10M/k100/efS750_qs10000_r_ge_0.90.txt"

# input_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k100/efS1000_qs10000.txt"
# output_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k100/efS1000_qs10000_r_ge_0.90.txt"

input_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/T2I1M/k100/efS1000_qs10000_full.txt"
output_file = "/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/T2I1M/k100/efS1000_qs10000_r_ge_0.90.txt"

df = pd.read_csv(input_file)

# 2. 过滤 r >= 0.90
df_filtered = df[df["r"] >= 0.90]

# 3. 保存为新文件（保持 CSV 格式）
df_filtered.to_csv(output_file, index=False)

print(f"原始数据量: {len(df)}")
print(f"筛选后数据量: {len(df_filtered)}")
