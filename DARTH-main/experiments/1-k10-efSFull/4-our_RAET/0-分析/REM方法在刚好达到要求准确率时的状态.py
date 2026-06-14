import csv

count = 0
# with open('/root/DARTH-main-data/sift-搜索刚好达到95准确率时的状态.csv', 'r', newline='', encoding='utf-8') as f:
with open('/root/DARTH-main-data/sift-CNS500-搜索刚好达到99准确率时的状态.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # 确保行至少有3列
        if len(row) >= 3:
            if row[1] < row[2]:
                count += 1

print("第二列和第三列值不同的行数:", count)