import os
import json

from tqdm import tqdm

import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import dask.dataframe as dd

def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def compute_P1(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 1)

dataset_params = {
        "SIFT1M": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d":128
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
        },
        "DEEP100M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
        },
        "T2I100M": {
            "M": 80,
            "efC": 1000,
            "efS": 2500,
        },
    }

SEED = 42

n_estimators = 100

qid_feat=["qid"]
#索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
index_metric_feats = ["step", "dists", "inserts","visited_points"]
#近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
#近邻统计特征，包括（近邻距离的平均值，方差，25，50，75分位数）
neighbor_stats_feats = ["avg_dist", "variance"]
all_feats = qid_feat+ index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

model_conf = {
    "darth-lightgbm": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
}

feature_classes = {
    #"index_metric_feats": index_metric_feats,
    #"neighbor_distances_feats": neighbor_distances_feats,
    #"neighbor_stats_feats": neighbor_stats_feats,
    #"index_metrics_and_neighbor_distances": index_metric_feats + neighbor_distances_feats,
    #"index_metrics_and_neighbor_stats": index_metric_feats + neighbor_stats_feats,
    #"neighbor_distances_and_neighbor_stats": neighbor_distances_feats + neighbor_stats_feats,
    "all_feats": all_feats,
}
def read_fvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected fvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = (dim + 1) * 4  # 4 bytes per float

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        num_vectors = file_size // vector_size

        #print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")

        if limit is not None and limit < num_vectors:
            num_vectors = limit
            #print(f">>Limiting to {num_vectors} vectors")

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)

        vectors = data[:, 1:]

        #print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)

    return vectors, int(dim)
def get_validation_dims_df(data_df, d, ds_name, s):
    dimensions = np.arange(1, d + 1)
    per_query_dimensions_df = pd.DataFrame(index=data_df["qid"].unique(), columns=[f"d{d}" for d in dimensions])

    query_dims = None
    dims_read = None

    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ann-data/sift/sift_query.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs(f"/data/mchatzakis/datasets/processed/GIST1M/learn.100K.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs(f"/data/mchatzakis/datasets/processed/GLOVE100/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP100M":
        query_dims, dims_read = read_fvecs(f"/data/mchatzakis/datasets/processed/DEEP100M/learn.1M.fvecs", limit=s)
    elif ds_name == "T2I100M":
        query_dims, dims_read = read_fvecs(f"/data/mchatzakis/datasets/processed/T2I100M/learn.1M.fvecs", limit=s)

    assert query_dims.shape[0] == s, f"Expected {s} queries, got {query_dims.shape[0]}"
    assert query_dims.shape[1] == d, f"Expected {d} dimensions, got {query_dims.shape[1]}"
    assert dims_read == d, f"Expected {d} dimensions, got {dims_read}"

    # Make sure that this is correct
    for i, qid in enumerate(data_df["qid"].unique()):
        per_query_dimensions_df.loc[qid] = query_dims[i]

        #assert len(per_query_dimensions_df) == s, f"Expected {s} queries, got {len(per_query_dimensions_df)}"

    return per_query_dimensions_df, dimensions
def get_validation_dataset_name(M, efC, efS, query_num, ds_name, k, logint): 
    return f"/home/extra_home/lxx23125236/AKNNS_experiment/DARTH-main/data/et_training_data/test_logging/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{query_num}_li{logint}_visitedPoints.txt"

results = {}
# file_path = "predictor_validation_results.json"
#
# if os.path.exists(file_path):
#     with open(file_path, "r") as f:
#         results = json.load(f)
# else:
#     print(f"File '{file_path}' does not exist.")
    
    
training_queries_num = ["10000"]
all_k_values = ["100"]
all_datasets = ["SIFT1M"]
all_lis = ["1"]

for ds_name in all_datasets:
    if ds_name not in results:
        results[ds_name] = {}
    
    for k in tqdm(all_k_values, desc=f"k values for {ds_name}"):
        if k not in results[ds_name]:
            results[ds_name][k] = {}
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        
        validation_data_dd = dd.read_csv(get_validation_dataset_name(M, efC, efS, 1000, ds_name, k, 1), usecols=columns_to_load)
        validation_data_df = validation_data_dd.compute()
        validation_y_true = validation_data_df["r"]
        per_validation_dimensions_df, dimensions = get_validation_dims_df(validation_data_df, d, ds_name, 1000)

        for s in training_queries_num:
            if s not in results[ds_name][k]:
                results[ds_name][k][s] = {}
            
            for li in all_lis:
                if li not in results[ds_name][k][s]:
                    results[ds_name][k][s][li] = {}
                
                for selected_features in feature_classes.keys():
                    
                    feats = feature_classes[selected_features]
                    
                    model_file = f"../predictor_models/raet/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}.txt"
                    model = lgb.Booster(model_file=model_file)
                    
                    validation_X = validation_data_df[feats]

                    validation_X = validation_X.merge(per_validation_dimensions_df, left_on="qid", right_index=True)
                    query_dims_features = [f"d{d}" for d in dimensions]
                    input_features =  ["step", "dists", "inserts","visited_points","first_nn_dist", "nn_dist", "furthest_dist","avg_dist", "variance"] + query_dims_features
                    validation_X = validation_X[input_features].values


                    validation_y_pred = model.predict(validation_X)
                    mse = mean_squared_error(validation_y_true, validation_y_pred)
                    mae = mean_absolute_error(validation_y_true, validation_y_pred)
                    r2 = r2_score(validation_y_true, validation_y_pred)
                    p99 = compute_P99(validation_y_true, validation_y_pred)
                    p1 = compute_P1(validation_y_true, validation_y_pred)
                    
                    #average_relative_error = np.mean(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
                    #maximum_relative_error = np.max(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
                    
                    results[ds_name][k][s][li][selected_features] = {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "p99": p99,
                        "p1": p1,
                        #"average_relative_error": average_relative_error,
                        #"maximum_relative_error": maximum_relative_error
                    }
                    print(json.dumps(results, indent=2, default=float))
                    # "mse": 0.0008999710665506829,
                    # "mae": 0.020367973013562093,
                    # "r2": 0.9629585105358561,
                    # "p99": 0.11116456370569527,
                    # "p1": 0.00023773482949720925
                

# with open(file_path, "w") as f:
#     json.dump(results, f, indent=4)