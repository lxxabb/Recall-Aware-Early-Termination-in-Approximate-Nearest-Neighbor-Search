import os
import time
import subprocess
import json
from sys import stderr

import pandas as pd
import numpy as np
import re

# LAET Tuning. Memory: False, training Size: 1000. Multiplier range: [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96, 1.01, 1.06, 1.11, 1.16, 1.21, 1.26, 1.31, 1.36, 1.41, 1.46, 1.51, 1.56, 1.61, 1.66, 1.71, 1.76, 1.81, 1.86, 1.91, 1.96, 2.01, 2.06, 2.11, 2.16, 2.21, 2.26, 2.31, 2.36, 2.41, 2.46, 2.51, 2.56, 2.61, 2.66, 2.71, 2.76, 2.81, 2.86, 2.91, 2.96, 3.01, 3.06, 3.11, 3.16, 3.21, 3.26, 3.31, 3.36, 3.41, 3.46, 3.51, 3.56, 3.61, 3.66, 3.71, 3.76, 3.81, 3.86, 3.91, 3.96, 4.01, 4.06, 4.11, 4.16, 4.21, 4.26, 4.31, 4.36, 4.41, 4.46, 4.51, 4.56, 4.61, 4.66, 4.71, 4.76, 4.81, 4.86, 4.91, 4.96, 5.01]
# Dataset: T2I1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.01, Average Recall: 0.969469, Tuning Time: 35.19707 ms, Total Experiments: 6
# Dataset: T2I1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.01, Average Recall: 0.969469, Tuning Time: 30.65281 ms, Total Experiments: 6
# Dataset: T2I1M, k: 100, R_t: 0.97, Minimum Multiplier: 0.26, Average Recall: 0.970609, Tuning Time: 41.969170000000005 ms, Total Experiments: 7
# Dataset: T2I1M, k: 100, R_t: 0.98, Minimum Multiplier: 0.56, Average Recall: 0.980918, Tuning Time: 42.07922 ms, Total Experiments: 7
# Dataset: T2I1M, k: 100, R_t: 0.99, Minimum Multiplier: 1.01, Average Recall: 0.990208, Tuning Time: 41.80737 ms, Total Experiments: 7

# Dataset: SIFT1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.81, Average Recall: 0.95906, Tuning Time: 40.982229999999994 ms, Total Experiments: 7
# Dataset: SIFT1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.86, Average Recall: 0.966443, Tuning Time: 40.65226 ms, Total Experiments: 7
# Dataset: SIFT1M, k: 100, R_t: 0.97, Minimum Multiplier: 0.91, Average Recall: 0.972326, Tuning Time: 37.85147 ms, Total Experiments: 6
# Dataset: SIFT1M, k: 100, R_t: 0.98, Minimum Multiplier: 1.01, Average Recall: 0.980662, Tuning Time: 43.52441 ms, Total Experiments: 7
# Dataset: SIFT1M, k: 100, R_t: 0.99, Minimum Multiplier: 1.26, Average Recall: 0.991295, Tuning Time: 48.450770000000006 ms, Total Experiments: 6
# Dataset: GIST1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.76, Average Recall: 0.950938, Tuning Time: 42.16268 ms, Total Experiments: 6
# Dataset: GIST1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.86, Average Recall: 0.961898, Tuning Time: 48.34326 ms, Total Experiments: 7
# Dataset: GIST1M, k: 100, R_t: 0.97, Minimum Multiplier: 1.01, Average Recall: 0.972348, Tuning Time: 50.04735 ms, Total Experiments: 7
# Dataset: GIST1M, k: 100, R_t: 0.98, Minimum Multiplier: 1.21, Average Recall: 0.981098, Tuning Time: 51.69317 ms, Total Experiments: 7
# Dataset: GIST1M, k: 100, R_t: 0.99, Minimum Multiplier: 1.66, Average Recall: 0.990278, Tuning Time: 63.17631 ms, Total Experiments: 7
# Dataset: GLOVE100, k: 100, R_t: 0.95, Minimum Multiplier: 0.86, Average Recall: 0.951892, Tuning Time: 17.08602 ms, Total Experiments: 7
# Dataset: GLOVE100, k: 100, R_t: 0.96, Minimum Multiplier: 0.91, Average Recall: 0.966394, Tuning Time: 15.58644 ms, Total Experiments: 6
# Dataset: GLOVE100, k: 100, R_t: 0.97, Minimum Multiplier: 0.96, Average Recall: 0.976301, Tuning Time: 17.91582 ms, Total Experiments: 7
# Dataset: GLOVE100, k: 100, R_t: 0.98, Minimum Multiplier: 1.01, Average Recall: 0.98294, Tuning Time: 18.06226 ms, Total Experiments: 7
# Dataset: GLOVE100, k: 100, R_t: 0.99, Minimum Multiplier: 1.16, Average Recall: 0.990883, Tuning Time: 18.48382 ms, Total Experiments: 7
# Dataset: DEEP10M, k: 100, R_t: 0.95, Minimum Multiplier: 0.96, Average Recall: 0.950569, Tuning Time: 68.49429 ms, Total Experiments: 7
# Dataset: DEEP10M, k: 100, R_t: 0.96, Minimum Multiplier: 1.06, Average Recall: 0.962278, Tuning Time: 71.1075 ms, Total Experiments: 7
# Dataset: DEEP10M, k: 100, R_t: 0.97, Minimum Multiplier: 1.16, Average Recall: 0.971153, Tuning Time: 71.39347000000001 ms, Total Experiments: 7
# Dataset: DEEP10M, k: 100, R_t: 0.98, Minimum Multiplier: 1.31, Average Recall: 0.980189, Tuning Time: 86.9227 ms, Total Experiments: 7
# Dataset: DEEP10M, k: 100, R_t: 0.99, Minimum Multiplier: 1.66, Average Recall: 0.990291, Tuning Time: 92.48509999999999 ms, Total Experiments: 7

dataset_info = {
    "SIFT1M":{
        "M": 32,
        "efC": 500,
        "efS": 500,
        "d": 128,
        "F": 741
    },
    "GIST1M": {
        "M": 32,
        "efC": 500,
        "efS": 1000,
        "d": 960,
        "F": 2260,
    },
    "GLOVE100": {
        "M": 16,
        "efC": 500,
        "efS": 500,
        "d": 100,
        "F": 700,
    },
    "DEEP10M": {
        "M": 32,
        "efC": 500,
        "efS": 750,
        "d": 96,
        "F": 1118,
    },
    "T2I1M": {
        "M": 80,
        "efC": 1000,
        "efS": 1000,
        "d": 200,
        "F": 4000,
    },
}

# all_datasets = ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]
all_datasets = ["T2I1M"]

all_k_values = [100]
all_recall_targets = [0.95, 0.96, 0.97, 0.98, 0.99]



def get_average_recall(training_df):
    return training_df["r"].mean()

def get_total_query_search_time(training_df):
    return training_df["elaps_ms"].sum()
import re

def parse_time_and_recall(output: str):
    """
    从 C++ stdout 中提取：
    - search_time (double)
    - recall (float)
    """

    # 1️⃣ 提取搜索时间
    m_time = re.search(r"总搜索时间[:：]\s*([0-9.eE+-]+)", output)
    if not m_time:
        raise RuntimeError("Failed to parse search time")

    search_time = float(m_time.group(1))

    # 2️⃣ 提取准确率（支持中文 / 英文 / 带 K）
    m_recall = re.search(
        r"(?:\b\d+\s+)?NN\s+(?:Recall|准确率)\s*=\s*([0-9.eE+-]+)",
        output
    )
    if not m_recall:
        raise RuntimeError("Failed to parse recall")

    recall = float(m_recall.group(1))

    return recall, search_time


# LAET Automated Tuning
def run_laet(ds_name, M, efC, efS, s, F, k, m, target_recall, result_directory, output_log_file):
    mode = "Laet_test"
    os.makedirs(f"{result_directory}/{ds_name}/k{k}", exist_ok=True)
    m = float(m)   # 双保险
    command = [
        "/home/extra_home/lxx23125236/ali/NSG/cmake-build-lxx-release/tests/test_nsg_optimized_search",
        f"--dataset={ds_name}",
        f"--K={int(k)}",
        f"--L={int(efS)}",
        f"--mode={mode}",
        f"--query_type=train",
        f"--query_num={int(s)}",
        f"--target_recall={float(target_recall):.2f}",
        f"--laet_F={int(F)}",
        f"--laet_multiplier={m:.2f}",
        f"--model_path=/home/extra_home/lxx23125236/ali/NSG-data/predictor_models/laet/{ds_name}/k{int(k)}/efS{int(efS)}_s{10000}_f{int(F)}.txt"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    output = result.stdout + "\n" + result.stderr

    recall, search_time = parse_time_and_recall(output)

    return recall, search_time

def get_laet_training_result_filename(ds_name, M, efC, efS, s, k, m, target_recall, result_directory):
    return f"{result_directory}/{ds_name}/k{k}/efS{efS}_qs{s}_tr{target_recall:.2f}_multiplier{m}.txt"

def find_min_multiplier(ds_name, M, efC, efS, s, F, k, recall_target, multipliers, min_m_idx, result_directory, output_log_file):
    """
    Find the minimum multiplier that achieves the target recall using binary search,
    starting from a given minimum index.
    """
    low, high = min_m_idx, len(multipliers) - 1
    best_m = None  # To store the minimum multiplier that satisfies the condition

    total_experiments = 0
    total_tuning_time = 0

    while low <= high:
        mid = (low + high) // 2
        m = multipliers[mid]

        # Run the algorithm and retrieve the results
        avg_recall, search_time = run_laet(ds_name, M, efC, efS, s, F, k, m, recall_target, result_directory, output_log_file)
        # training_df = pd.read_csv(get_laet_training_result_filename(ds_name, M, efC, efS, s, k, m, recall_target, result_directory))
        # avg_recall = get_average_recall(training_df)

        if avg_recall >= recall_target:
            best_m = m
            high = mid - 1 # Search for smaller m
        else:
            low = mid + 1  # Search for larger m

        total_experiments += 1
        total_tuning_time += search_time

    if best_m is None:
        print(f"[WARNING] -- Dataset: {ds_name}, k: {k}, R_t: {recall_target}, No suitable multiplier found. Using max multiplier.")
        best_m = multipliers[-1]

    return best_m, low, total_experiments, total_tuning_time

def laet_tuning(s, keep_m_idx_memory = False,):
    result_directory = "/home/extra_home/lxx23125236/ali/NSG-data/results/Laet-达到要求准确率的最小参数"
    output_log_filename = "/home/extra_home/lxx23125236/ali/NSG-data/results/Laet-达到要求准确率的最小参数/k100_laet_tuning.log"

    output_log_file = open(output_log_filename, "w")

    step = 0.05
    multiplier_range_upper_bound = 5.0
    multiplier_range_lower_bound = 0.01
    multipliers = np.arange(multiplier_range_lower_bound, multiplier_range_upper_bound + step, step)
    # multipliers = [f"{m:.2f}" for m in multipliers]
    multipliers = np.round(multipliers, 2).tolist()   # 全是 float
    print(f"LAET Tuning. Memory: {keep_m_idx_memory}, training Size: {s}. Multiplier range: {multipliers}")

    tuning_results = {}
    for di, ds_name in enumerate(all_datasets):
        M = dataset_info[ds_name]["M"]
        F = dataset_info[ds_name]["F"]
        efS = dataset_info[ds_name]["efS"]
        efC = dataset_info[ds_name]["efC"]

        tuning_results[ds_name] = {}

        for k in all_k_values:
            tuning_results[ds_name][k] = {}
            min_m_idx = 0
            for recall_target in all_recall_targets:
                min_m, min_m_idx, total_experiments, total_tuning_time = find_min_multiplier(ds_name, M, efC, efS, s, F, k, recall_target, multipliers, min_m_idx, result_directory, output_log_file)

                # training_df = pd.read_csv(get_laet_training_result_filename(ds_name, M, efC, efS, s, k, min_m, recall_target, result_directory))
                # avg_recall = get_average_recall(training_df)
                avg_recall, search_time = run_laet(ds_name, M, efC, efS, s, F, k, min_m, recall_target, result_directory, output_log_file)

                print(f"Dataset: {ds_name}, k: {k}, R_t: {recall_target}, Minimum Multiplier: {min_m}, Average Recall: {avg_recall}, Tuning Time: {total_tuning_time} ms, Total Experiments: {total_experiments}")

                tuning_results[ds_name][k][recall_target] = {
                    "min_m": min_m,
                    "avg_recall": avg_recall,
                    "total_experiments": total_experiments,
                    "total_tuning_time": total_tuning_time
                }

                if not keep_m_idx_memory:
                    min_m_idx = 0

    filename = f"/home/extra_home/lxx23125236/ali/NSG-data/results/Laet-达到要求准确率的最小参数/k100_laet_tuning_results_memory{keep_m_idx_memory}_trainingSize{s}.json"
    with open(filename, "w") as f:
        json.dump(tuning_results, f, indent=4)

if __name__ == "__main__":
    for s in [1000]:
        laet_tuning(s, keep_m_idx_memory=False)
