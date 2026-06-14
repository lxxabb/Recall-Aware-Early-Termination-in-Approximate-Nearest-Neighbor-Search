import csv
input_file = "/root/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k10/efS600_qs10000.txt"
output_file = "/root/NSG-data/train_data/Laet_Darth_train_data/GIST1M/k10/efS600_qs10000_r_ge_0.9.txt"
# input_file = "/root/NSG-data/train_data/Laet_Darth_train_data/SIFT1M/k10/efS150_qs10000.txt"
# output_file = "/root/NSG-data/train_data/Laet_Darth_train_data/SIFT1M/k10/efS150_qs10000_r_ge_0.9.txt"
# input_file = "/root/NSG-data/train_data/Laet_Darth_train_data/DEEP10M/k10/efS300_qs10000.txt"
# output_file = "/root/NSG-data/train_data/Laet_Darth_train_data/DEEP10M/k10/efS300_qs10000_r_ge_0.9.txt"
threshold = 0.9

with open(input_file, "r", newline="") as fin, \
        open(output_file, "w", newline="") as fout:

    reader = csv.DictReader(fin)   # 按列名读
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)

    # 写 header
    writer.writeheader()

    # 只保留 r >= 0.9
    for row in reader:
        r = float(row["r"])
        if r >= threshold:
            writer.writerow(row)
