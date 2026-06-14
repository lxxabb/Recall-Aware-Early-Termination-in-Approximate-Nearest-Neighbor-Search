import csv
import sys
def main(filename, target_r=0.90):
    first_dists_at_target = {}

    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            # 处理 UTF-8 BOM（如果存在）
            header = f.readline()
            if header.startswith('\ufeff'):
                f.seek(0)
                f.read(1)  # 跳过 BOM
            else:
                f.seek(0)

            reader = csv.DictReader(f)
            required_fields = {'qid', 'dists', 'r'}
            if not required_fields.issubset(reader.fieldnames):
                print("错误：文件缺少必要字段。请确保包含: qid, dists, r")
                print("实际字段:", reader.fieldnames)
                return

            for row in reader:
                try:
                    qid = int(row['qid'])
                    dists = int(row['dists'])      # 若 dists 是浮点数，可改为 float
                    r_val = float(row['r'])
                except (ValueError, KeyError) as e:
                    continue  # 静默跳过无效行（或可加警告）

                if r_val >= target_r and qid not in first_dists_at_target:
                    first_dists_at_target[qid] = dists

        if not first_dists_at_target:
            print(f"没有查询达到 r >= {target_r:.4f}")
            return

        avg_dists = sum(first_dists_at_target.values()) / len(first_dists_at_target)
        print(f"目标准确率: r >= {target_r:.4f}")
        print(f"成功达到目标的查询数量: {len(first_dists_at_target)}")
        print(f"平均距离计算次数 (dists): {avg_dists:.2f}")
        print(f"ipi: {avg_dists/2:.0f}")
        print(f"mpi: {avg_dists/10:.0f}")
        print("===============================")


    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

def parse_args():
    if len(sys.argv) < 2:
        print("用法: python compute_avg_dists.py <logfile.txt> [target_r]")
        print("示例: python compute_avg_dists.py log.txt 0.95")
        sys.exit(1)

    logfile = sys.argv[1]
    target_r = 0.90  # 默认值

    if len(sys.argv) >= 3:
        try:
            target_r = float(sys.argv[2])
            if not (0.0 <= target_r <= 1.0):
                print("警告：target_r 应在 [0.0, 1.0] 范围内，但程序仍会继续运行。")
        except ValueError:
            print(f"错误：无法解析 target_r 参数 '{sys.argv[2]}' 为浮点数。")
            sys.exit(1)

    return logfile, target_r

if __name__ == "__main__":
    logfile, target_r = parse_args()
    main(logfile, target_r)