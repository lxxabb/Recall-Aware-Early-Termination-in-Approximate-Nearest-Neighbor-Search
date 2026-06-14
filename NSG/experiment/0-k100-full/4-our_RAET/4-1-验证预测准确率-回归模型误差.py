import os

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import json

from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd

def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def compute_P1(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 1)

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    # return f"/root/DARTH-main-data/data/et_training_data/raet-fea-testing/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
    return f"/root/NSG-data/test_data/Laet_Darth_train_data/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"

def main():
    SEED = 42
    n_estimators = 100
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,visited_points,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
    qid_feat=["qid"]
    #索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
    index_metric_feats = ["step", "dists", "inserts","visited_points","duration"]
    #近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    #近邻统计特征，包括（近邻距离的平均值，方差，密度特征（平均距离的倒数）,入口点到查询点距离 与入口点距离均值之间的比值特征,距离分布熵 ,）
    neighbor_stats_feats = ["mean_distance", "variance_of_distances"]

    all_feats = qid_feat+ index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

    columns_to_load = all_feats + ["r"]

    model_conf = {
        "raet-cns": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
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

    dataset_params = {
        "SIFT1M": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d":128,
            "query_num":10000
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
            "query_num":1000
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "query_num":10000
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
            "query_num":10000
        },
    }

    training_queries_num = [10000]
    all_k_values = [100]
    all_datasets = ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]
    all_li = [1]
    target_recalls = [0.95,0.96,0.97,0.98,0.99]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        query_num = dataset_params[ds_name]["query_num"]
        results = train_predictors(
            ds_name, query_num, all_k_values, all_li,
            M, efC, efS, d,n_estimators, columns_to_load, feature_classes, model_conf,target_recalls)

        # with open(f"/root/DARTH-main-data/predictor_models/raet-cns-regression/{ds_name}_qs10000.json", "w") as f:
        #     json.dump(results, f, indent=4)

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
def get_query_dims_df(data_df, d, ds_name, s):
    dimensions = np.arange(1, d + 1)
    per_query_dimensions_df = pd.DataFrame(index=data_df["qid"].unique(), columns=[f"d{d}" for d in dimensions])
    """
    qid_df：必须是 pandas DataFrame，只包含 qid 列
    """

    # ---- 先读 query embedding ----
    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/SIFT1M/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/GIST1M/gist_learn.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/GLOVE100/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP10M":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs", limit=s)

    assert query_dims.shape[0] == s, f"Expected {s} queries, got {query_dims.shape[0]}"
    assert query_dims.shape[1] == d, f"Expected {d} dimensions, got {query_dims.shape[1]}"
    assert dims_read == d, f"Expected {d} dimensions, got {dims_read}"

    # Make sure that this is correct
    for i, qid in enumerate(data_df["qid"].unique()):
        per_query_dimensions_df.loc[qid] = query_dims[i]

        #assert len(per_query_dimensions_df) == s, f"Expected {s} queries, got {len(per_query_dimensions_df)}"

    return per_query_dimensions_df, dimensions

def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls):
    results = {}
    train_num=10000
    print(f"开始进行模型预测 for {ds_name}")
    s=training_queries_num
    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, s, ds_name, k, li)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        query_data_all = dd.read_csv(data_path, usecols=columns_to_load,dtype={'r': 'float64'})
        # all_queries_data = all_queries_dask.compute()

        results[k][s] = {}

        # compute qid df
        qid_pd = query_data_all[["qid"]].compute()

        # load query vectors
        # per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)

        y_train = query_data_all["r"].compute()

        for li in all_li:
            results[k][s][li] = {}

            # filter + compute (IMPORTANT FIX)
            data_all = query_data_all[query_data_all["dists"] % li == 0].compute()

            print(f"    {s} Queries Data Shape: {data_all.shape} | Li: {li}")

            assert len(data_all) > 0, "data_all empty after merge — check labels/qid alignment"

            for selected_features in feature_classes:
                feats = feature_classes[selected_features]

                # data_df = data_all[feats].merge(
                #     per_query_dimensions_df,
                #     left_on="qid",
                #     right_index=True,
                #     how="left"
                # )

                # query_dims_features = [f"d{i}" for i in dimensions]

                input_features =  ["step", "dists", "inserts","visited_points","duration","nn_dist","furthest_dist","mean_distance","variance_of_distances"]
                data_all["visited_points"] = data_all["visited_points"]/k
                X_train = data_all[input_features].values

                model_file = f"/root/NSG-data/predictor_models/raet/{ds_name}/k{k}/efS{efS}_s10000_nestim{n_estimators}_{selected_features}_Noquery_duration.txt"
                model = lgb.Booster(model_file=model_file)
                y_pred = model.predict(X_train)

                mse = mean_squared_error(y_train, y_pred)
                mae = mean_absolute_error(y_train, y_pred)
                r2 = r2_score(y_train, y_pred)
                p99 = compute_P99(y_train, y_pred)
                p1 = compute_P1(y_train, y_pred)

                results[k][s][li][selected_features] = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "p99": p99,
                    "p1": p1,
                }
        print(results)

    print(f"Finished predict for {ds_name}")
    print("\n")

    return results


if __name__ == "__main__":
    main()

# Starting training for SIFT1M
#     SIFT1M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_train_data/SIFT1M/k100/efS500_qs10000.txt
# 10000 Queries Data Shape: (6500321, 12) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 0.005153296296218798, 'mae': 0.05369705235338278, 'r2': 0.9432330057717913, 'p99': 0.20748168677981704, 'p1': 0.0006157299620752621}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_train_data/GIST1M/k100/efS1000_qs1000.txt
# 1000 Queries Data Shape: (796942, 12) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 0.010135261312369193, 'mae': 0.07430054489311687, 'r2': 0.8840837567659257, 'p99': 0.2990340959380769, 'p1': 0.0008185235404958718}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_train_data/GLOVE100/k100/efS500_qs10000.txt
# 10000 Queries Data Shape: (4514780, 12) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 0.0025965987237661363, 'mae': 0.03498013226395755, 'r2': 0.9706974172393377, 'p99': 0.1666687941016436, 'p1': 0.0004491597172320283}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_train_data/DEEP10M/k100/efS750_qs10000.txt
# 10000 Queries Data Shape: (6076582, 12) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 0.007037283057682502, 'mae': 0.061866375363964186, 'r2': 0.9201150928076911, 'p99': 0.2476357025414258, 'p1': 0.0007952996804774237}}}}}
# Finished predict for DEEP10M
