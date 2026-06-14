# Dataset: SIFT1M, k: 100, R_t: 0.95, Minimum efSearch: 100, Average Recall: 0.97174, Tuning Time: 6228.312496 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.96, Minimum efSearch: 100, Average Recall: 0.97174, Tuning Time: 6224.833077999999 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.97, Minimum efSearch: 100, Average Recall: 0.97174, Tuning Time: 6219.052804 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.98, Minimum efSearch: 117, Average Recall: 0.9804100000000001, Tuning Time: 6756.485464000001 ms, Total Experiments: 10
# Dataset: SIFT1M, k: 100, R_t: 0.99, Minimum efSearch: 155, Average Recall: 0.99, Tuning Time: 7628.802784999999 ms, Total Experiments: 10
# Dataset: GLOVE100, k: 100, R_t: 0.95, Minimum efSearch: 100, Average Recall: 0.97306, Tuning Time: 2245.8818140000003 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.96, Minimum efSearch: 100, Average Recall: 0.97306, Tuning Time: 2233.3856349999996 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.97, Minimum efSearch: 100, Average Recall: 0.97306, Tuning Time: 2261.593649 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.98, Minimum efSearch: 128, Average Recall: 0.98, Tuning Time: 2400.3200419999994 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.99, Minimum efSearch: 299, Average Recall: 0.99001, Tuning Time: 4407.884016 ms, Total Experiments: 10
# Dataset: GIST1M, k: 100, R_t: 0.95, Minimum efSearch: 246, Average Recall: 0.95001, Tuning Time: 28555.192023000003 ms, Total Experiments: 10
# Dataset: GIST1M, k: 100, R_t: 0.96, Minimum efSearch: 287, Average Recall: 0.96016, Tuning Time: 30745.797639999997 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.97, Minimum efSearch: 348, Average Recall: 0.97, Tuning Time: 37323.900465000006 ms, Total Experiments: 10
# Dataset: GIST1M, k: 100, R_t: 0.98, Minimum efSearch: 452, Average Recall: 0.98, Tuning Time: 44996.54123300001 ms, Total Experiments: 10
# Dataset: GIST1M, k: 100, R_t: 0.99, Minimum efSearch: 695, Average Recall: 0.9900600000000002, Tuning Time: 55794.68321100001 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.95, Minimum efSearch: 100, Average Recall: 0.95401, Tuning Time: 8027.258900000001 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.96, Minimum efSearch: 111, Average Recall: 0.96006, Tuning Time: 8697.065107 ms, Total Experiments: 10
# Dataset: DEEP10M, k: 100, R_t: 0.97, Minimum efSearch: 134, Average Recall: 0.9701900000000001, Tuning Time: 9285.677216000002 ms, Total Experiments: 10
# Dataset: DEEP10M, k: 100, R_t: 0.98, Minimum efSearch: 174, Average Recall: 0.98022, Tuning Time: 9770.373367 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.99, Minimum efSearch: 259, Average Recall: 0.99002, Tuning Time: 13116.596927 ms, Total Experiments: 10

import os
import time
import subprocess
import json
import pandas as pd
import numpy as np

INDEX_DIRECTORY = "/root/DARTH-main-data/data/HNSW-index"
DATASET_DIRECTORY = "/root/ann-data/"

dataset_info = {
        "SIFT1M":{
            "M": 32,
            "efC": 500,
            "efS": 500,
             "d": 128,
             "F": 241 #固定 early termination search 预算
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
            "F": 1260,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "F": 200,
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
            "F": 368,
        },
}

all_datasets = ["SIFT1M","GLOVE100","GIST1M","DEEP10M"]
# all_k_values = [10, 25, 50, 75, 100]
# all_recall_targets = [0.80, 0.85, 0.90, 0.95,]

all_k_values = [100]
all_recall_targets = [0.95, 0.96, 0.97, 0.98, 0.99]

#计算召回率均值
def get_average_recall(training_df):
    # print(repr(training_df.columns.tolist()))
    # print(training_df["r"].head())
    # print(training_df["r"].dtype)
    # print(training_df["r"].isna().sum())

    return training_df["r"].mean()
#统计总查询时间 ms
def get_total_query_search_time(training_df):
    return training_df["elaps_ms"].sum()


# Classic HNSW without Early Termination efSearch Tuning
def run_classic_hnsw(ds_name, M, efC, efS, s, k, target_recall, result_directory, output_log_file):
    mode = "no-early-stop"
    os.makedirs(f"{result_directory}/{ds_name}/k{k}", exist_ok=True)
    
    command = [
        "/root/DARTH-main/cmake-build-lxx-release/hnsw-test/hnsw_test",
        "--dataset", ds_name,
        "--M", str(M),
        "--efConstruction", str(efC),
        "--efSearch", str(efS),
        "--query-num", str(s),
        "--k", str(k),
        "--output", f"{result_directory}/{ds_name}/k{k}/efS{efS}_qs{s}_tr{target_recall:.2f}.txt",
        "--mode", mode,
        "--index-filepath", f"{INDEX_DIRECTORY}/{ds_name}/{ds_name}.M{M}.efC{efC}.index",
        "--dataset-dir-prefix", DATASET_DIRECTORY,
        "--target-recall", f"{target_recall:.2f}",
        "--query-type", "training",
    ]
    
    try:
        #print(f"Running HNSW with efSearch: {efS}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_log_file.write(f"{result.stdout}\n\n")
        if result.stderr != "":
            print("Command Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code:", e.returncode)
        print("Error Output:", e.stderr)

def get_classic_hnsw_training_result_filename(ds_name, M, efC, efS, s, k, target_recall, result_directory):
    return f"{result_directory}/{ds_name}/k{k}/efS{efS}_qs{s}_tr{target_recall:.2f}.txt"

def find_best_efSearch(ds_name, M, efC, s, k, recall_target, efSearch_values, min_efS_idx, result_directory, output_log_file):
    low, high = min_efS_idx, len(efSearch_values) - 1
    best_efS = None
    
    total_experiments = 0
    total_tuning_time = 0
    
    while low <= high:
        mid = (low + high) // 2
        efS = efSearch_values[mid]
        
        run_classic_hnsw(ds_name, M, efC, efS, s, k, recall_target, result_directory, output_log_file)
        training_df = pd.read_csv(get_classic_hnsw_training_result_filename(ds_name, M, efC, efS, s, k, recall_target, result_directory))
        avg_recall = get_average_recall(training_df)
        
        if avg_recall >= recall_target:
            best_efS = efS
            high = mid - 1
        else:
            low = mid + 1
        
        total_experiments += 1
        total_tuning_time += get_total_query_search_time(training_df)
        
    if best_efS is None:
        print(f"[WARNING] -- Dataset: {ds_name}, k: {k}, R_t: {recall_target}, No suitable efSearch found. Using max efSearch.")
        best_efS = efSearch_values[-1]
        
    return best_efS, low, total_experiments, total_tuning_time

def classic_hnsw_tuning(s, keep_efS_idx_memory = False):
    result_directory = "/root/DARTH-main-data/data/result/达到要求准确率的最小CNS"
    output_log_file = open("/root/DARTH-main-data/data/result/达到要求准确率的最小CNS/classic_hnsw_tuning.log", "w")
    
    step = 1
    efSearch_upper_bound = 1000
    efSearch_lower_bound = 100
    efSearch_values = np.arange(efSearch_lower_bound, efSearch_upper_bound + step, step)
    # make efSearch values ints
    efSearch_values = [int(efS) for efS in efSearch_values]
    print(f"efSearch range: {efSearch_values}")
    
    tuning_results = {}
    for di, ds_name in enumerate(all_datasets):
        M = dataset_info[ds_name]["M"]
        F = dataset_info[ds_name]["F"]
        efC = dataset_info[ds_name]["efC"]
        
        tuning_results[ds_name] = {}
        for k in all_k_values:
            tuning_results[ds_name][k] = {}
            min_efS_idx = 0
            for recall_target in all_recall_targets:
                
                min_efS, min_efS_idx, total_experiments, total_tuning_time = find_best_efSearch(ds_name, M, efC, s, k, recall_target, efSearch_values, min_efS_idx, result_directory, output_log_file)
                
                training_df = pd.read_csv(get_classic_hnsw_training_result_filename(ds_name, M, efC, min_efS, s, k, recall_target, result_directory))
                avg_recall = get_average_recall(training_df)
                    
                print(f"Dataset: {ds_name}, k: {k}, R_t: {recall_target}, Minimum efSearch: {min_efS}, Average Recall: {avg_recall}, Tuning Time: {total_tuning_time} ms, Total Experiments: {total_experiments}")
                    
                tuning_results[ds_name][k][recall_target] = {
                    "min_efS": min_efS,
                    "avg_recall": avg_recall,
                    "total_experiments": total_experiments,
                    "total_tuning_time": total_tuning_time
                }
                if not keep_efS_idx_memory:
                    min_efS_idx = 0
    
    filename = f"/root/DARTH-main-data/data/result/达到要求准确率的最小CNS/classic_hnsw_tuning_results_memory{keep_efS_idx_memory}_trainingSize{s}.json"
    with open(filename, "w") as f:
        json.dump(tuning_results, f, indent=4)

#keep_efS_idx_memory：作用是：是否在下一个 recall target 中
#False 每次 recall 从 100 重新搜
#True 下一次 recall 直接从“上一次找到的位置附近”开始二分
if __name__ == "__main__":
    for s in [1000]:
        classic_hnsw_tuning(s, keep_efS_idx_memory=True)