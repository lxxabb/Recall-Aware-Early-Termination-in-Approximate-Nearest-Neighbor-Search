import os
import time
import subprocess
import json
import pandas as pd
import numpy as np

# LAET Tuning. Memory: False, training Size: 1000. Multiplier range: ['0.01', '0.06', '0.11', '0.16', '0.21', '0.26', '0.31', '0.36', '0.41', '0.46', '0.51', '0.56', '0.61', '0.66', '0.71', '0.76', '0.81', '0.86', '0.91', '0.96', '1.01', '1.06', '1.11', '1.16', '1.21', '1.26', '1.31', '1.36', '1.41', '1.46', '1.51', '1.56', '1.61', '1.66', '1.71', '1.76', '1.81', '1.86', '1.91', '1.96', '2.01', '2.06', '2.11', '2.16', '2.21', '2.26', '2.31', '2.36', '2.41', '2.46', '2.51', '2.56', '2.61', '2.66', '2.71', '2.76', '2.81', '2.86', '2.91', '2.96', '3.01', '3.06', '3.11', '3.16', '3.21', '3.26', '3.31', '3.36', '3.41', '3.46', '3.51', '3.56', '3.61', '3.66', '3.71', '3.76', '3.81', '3.86', '3.91', '3.96', '4.01', '4.06', '4.11', '4.16', '4.21', '4.26', '4.31', '4.36', '4.41', '4.46', '4.51', '4.56', '4.61', '4.66', '4.71', '4.76', '4.81', '4.86', '4.91', '4.96', '5.01']
# Dataset: T2I1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.56, Average Recall: 0.9529099999999999, Tuning Time: 21527.340427 ms, Total Experiments: 7
# Dataset: T2I1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.61, Average Recall: 0.96116, Tuning Time: 24651.313321 ms, Total Experiments: 6
# Dataset: T2I1M, k: 100, R_t: 0.97, Minimum Multiplier: 0.71, Average Recall: 0.9716100000000001, Tuning Time: 23926.17039 ms, Total Experiments: 7
# Dataset: T2I1M, k: 100, R_t: 0.98, Minimum Multiplier: 0.86, Average Recall: 0.98145, Tuning Time: 22549.145720999997 ms, Total Experiments: 7
# Dataset: T2I1M, k: 100, R_t: 0.99, Minimum Multiplier: 1.11, Average Recall: 0.99003, Tuning Time: 27391.861201 ms, Total Experiments: 7

# Dataset: SIFT1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.55, Average Recall: 0.95221, Tuning Time: 5593.834903999999 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.59, Average Recall: 0.96059, Tuning Time: 5506.6249179999995 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.97, Minimum Multiplier: 0.65, Average Recall: 0.9701500000000001, Tuning Time: 5657.039401 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.98, Minimum Multiplier: 0.74, Average Recall: 0.98021, Tuning Time: 6581.104248 ms, Total Experiments: 9
# Dataset: SIFT1M, k: 100, R_t: 0.99, Minimum Multiplier: 0.91, Average Recall: 0.99049, Tuning Time: 7050.996801 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.95, Minimum Multiplier: 0.53, Average Recall: 0.95138, Tuning Time: 35504.902772 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.96, Minimum Multiplier: 0.59, Average Recall: 0.96023, Tuning Time: 37483.898855 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.97, Minimum Multiplier: 0.69, Average Recall: 0.9709800000000001, Tuning Time: 38761.13983 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.98, Minimum Multiplier: 0.83, Average Recall: 0.98053, Tuning Time: 47204.43580200001 ms, Total Experiments: 9
# Dataset: GIST1M, k: 100, R_t: 0.99, Minimum Multiplier: 1.07, Average Recall: 0.9902000000000001, Tuning Time: 54418.736456 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.95, Minimum Multiplier: 0.70, Average Recall: 0.95248, Tuning Time: 1984.0839270000001 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.96, Minimum Multiplier: 0.74, Average Recall: 0.96151, Tuning Time: 2318.594718 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.97, Minimum Multiplier: 0.79, Average Recall: 0.9703200000000001, Tuning Time: 2355.917949 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.98, Minimum Multiplier: 0.87, Average Recall: 0.9801300000000001, Tuning Time: 2431.2586629999996 ms, Total Experiments: 9
# Dataset: GLOVE100, k: 100, R_t: 0.99, Minimum Multiplier: 1.10, Average Recall: 0.9903, Tuning Time: 2698.0023149999997 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.95, Minimum Multiplier: 0.50, Average Recall: 0.9506400000000002, Tuning Time: 9227.47278 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.96, Minimum Multiplier: 0.56, Average Recall: 0.9616000000000001, Tuning Time: 9720.587200999998 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.97, Minimum Multiplier: 0.63, Average Recall: 0.97045, Tuning Time: 9989.787519000001 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.98, Minimum Multiplier: 0.74, Average Recall: 0.9802400000000001, Tuning Time: 11729.671467 ms, Total Experiments: 9
# Dataset: DEEP10M, k: 100, R_t: 0.99, Minimum Multiplier: 0.95, Average Recall: 0.9901700000000001, Tuning Time: 12720.156202000002 ms, Total Experiments: 9
INDEX_DIRECTORY = "/home/extra_home/lxx23125236/ali/DARTH-main-data/data/HNSW-index"
DATASET_DIRECTORY = "/home/extra_home/lxx23125236/ann-data/"

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
        "T2I1M": {
            "M": 80,
            "efC": 1000,
            "efS": 1000,
            "d": 200,
            "F": 400,
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


# LAET Automated Tuning
def run_laet(ds_name, M, efC, efS, s, F, k, m, target_recall, result_directory, output_log_file):
    mode = "laet-early-stop-testing"
    os.makedirs(f"{result_directory}/{ds_name}/k{k}", exist_ok=True)
    
    command = [
        "/home/extra_home/lxx23125236/ali/DARTH-main/cmake-build-lxx-release/hnsw-test/hnsw_test",
        "--dataset", ds_name,
        "--M", str(M),
        "--efConstruction", str(efC),
        "--efSearch", str(efS),
        "--query-num", str(s),
        "--k", str(k),
        "--output", f"{result_directory}/{ds_name}/k{k}/efS{efS}_qs{s}_tr{target_recall:.2f}_multiplier{m}.txt",
        "--mode", mode,
        "--index-filepath", f"{INDEX_DIRECTORY}/{ds_name}/{ds_name}.M{M}.efC{efC}.index",
        "--dataset-dir-prefix", DATASET_DIRECTORY,
        "--target-recall", f"{target_recall:.2f}",
        "--fixed-amount-of-search", str(F),
        "--prediction-multiplier", str(m),
        "--query-type", "training",
        "--predictor-model-path", f"/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/laet-our/{ds_name}/k{k}/efS{efS}_s{10000}_f{F}.txt"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_log_file.write(f"{result.stdout}\n\n")
        if result.stderr != "":
            output_log_file.write(f"{result.stderr}\n\n")
            print("Command Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code:", e.returncode)
        print("Error Output:", e.stderr)
    
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
        run_laet(ds_name, M, efC, efS, s, F, k, m, recall_target, result_directory, output_log_file)
        training_df = pd.read_csv(get_laet_training_result_filename(ds_name, M, efC, efS, s, k, m, recall_target, result_directory))
        avg_recall = get_average_recall(training_df)
        
        if avg_recall >= recall_target:
            best_m = m
            high = mid - 1 # Search for smaller m
        else:
            low = mid + 1  # Search for larger m
        
        total_experiments += 1
        total_tuning_time += get_total_query_search_time(training_df)
    
    if best_m is None:
        print(f"[WARNING] -- Dataset: {ds_name}, k: {k}, R_t: {recall_target}, No suitable multiplier found. Using max multiplier.")
        best_m = multipliers[-1]

    return best_m, low, total_experiments, total_tuning_time

def laet_tuning(s, keep_m_idx_memory = False,):
    result_directory = "/home/extra_home/lxx23125236/ali/DARTH-main-data/data/result/Laet-达到要求准确率的最小参数"
    output_log_filename = "/home/extra_home/lxx23125236/ali/DARTH-main-data/data/result/Laet-达到要求准确率的最小参数/laet_tuning.log"

    output_log_file = open(output_log_filename, "w") 
        
    step = 0.05
    multiplier_range_upper_bound = 5.0
    multiplier_range_lower_bound = 0.01
    multipliers = np.arange(multiplier_range_lower_bound, multiplier_range_upper_bound + step, step)
    multipliers = [f"{m:.2f}" for m in multipliers]
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
                
                training_df = pd.read_csv(get_laet_training_result_filename(ds_name, M, efC, efS, s, k, min_m, recall_target, result_directory))
                avg_recall = get_average_recall(training_df)
                    
                print(f"Dataset: {ds_name}, k: {k}, R_t: {recall_target}, Minimum Multiplier: {min_m}, Average Recall: {avg_recall}, Tuning Time: {total_tuning_time} ms, Total Experiments: {total_experiments}")
                    
                tuning_results[ds_name][k][recall_target] = {
                    "min_m": min_m,
                    "avg_recall": avg_recall,
                    "total_experiments": total_experiments,
                    "total_tuning_time": total_tuning_time
                } 

                if not keep_m_idx_memory:
                    min_m_idx = 0
    
    filename = f"/home/extra_home/lxx23125236/ali/DARTH-main-data/data/result/Laet-达到要求准确率的最小参数/laet_tuning_results_memory{keep_m_idx_memory}_trainingSize{s}.json"
    with open(filename, "w") as f:
        json.dump(tuning_results, f, indent=4)

if __name__ == "__main__":
    for s in [1000]:
        laet_tuning(s, keep_m_idx_memory=False)
