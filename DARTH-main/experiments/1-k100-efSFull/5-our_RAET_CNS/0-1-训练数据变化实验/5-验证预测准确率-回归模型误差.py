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

# def compute_P99(y_true, y_pred):
#     y_diff = np.abs(y_true - y_pred)
#     return np.percentile(y_diff, 99)

# def compute_P1(y_true, y_pred):
#     y_diff = np.abs(y_true - y_pred)
#     return np.percentile(y_diff, 1)

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/test/efS{efS}_qs{num_queries}.txt"
def get_label_dataset_name(ds_name, k,target_recall,num_queries,efS):
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/test/efs{efS}_min_ef_per_query_s{num_queries}_{target_recall}.csv"

def main():
    SEED = 42
    n_estimators = 100
    # n_estimators = 200
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,visited_points,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
    qid_feat=["qid"]
    #索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
    # index_metric_feats = ["dists", "inserts"]
    index_metric_feats = ["step", "dists", "inserts"]
    #近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    #近邻统计特征，包括（近邻距离的平均值，方差，密度特征（平均距离的倒数）,入口点到查询点距离 与入口点距离均值之间的比值特征,距离分布熵 ,）
    # neighbor_stats_feats = ["avg_dist","avg_hop","entry_query_dist_ratio","dist_distribution_entropy","kurtosis"]
    neighbor_stats_feats = ["avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy"]

    #余弦特征：
    # cosine_feats = ["cosine_mean"]
    cosine_feats = ["cosine_mean","cosine_variance","cosine_direction_entropy"]


    all_feats = qid_feat+ index_metric_feats + neighbor_distances_feats + neighbor_stats_feats + cosine_feats

    columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

    # model_conf = {
    #     "raet-cns": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
    # }


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
            "q":0.70,
            "query_num":10000
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
            "q":0.84,
            "query_num":1000
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "q":0.91,
            "query_num":10000
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
            "q":0.60,
            "query_num":10000
        },
    }

    training_nums = [1000, 5000, 10000,15000,20000]
    all_k_values = [100]
    k=all_k_values[0]
    all_datasets = ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]
    all_li = [1]
    target_recalls = [0.95,0.96,0.97,0.98,0.99]

    for ds_name in all_datasets:
        quantile=dataset_params[ds_name]["q"]
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        query_num = dataset_params[ds_name]["query_num"]
        for train_num in training_nums:
            results = train_predictors(
                ds_name, query_num, all_k_values, all_li,
                M, efC, efS, d,n_estimators, columns_to_load, feature_classes, quantile,target_recalls,train_num)

            # with open(f"/root/DARTH-main-data/predictor_models/raet-cns-regression/{ds_name}/k{k}/训练数据量_{train_num}_的误差.json", "w") as f:
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

def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, quantile,target_recalls,train_num):
    results = {}
    quantile=int(quantile*100)
    # train_num=10000
    print(f"开始验证 for {ds_name}")
    s=training_queries_num
    for k in all_k_values:
        for target_recall in target_recalls:
            results[k] = {}
            li = 1

            data_path = get_dataset_name(M, efC, efS, s, ds_name, k, li)

            print(f"{ds_name} | k={k} | Path: {data_path}")

            query_data_all = dd.read_csv(data_path, usecols=columns_to_load)
            # all_queries_data = all_queries_dask.compute()

            results[k][s] = {}

            # compute qid df
            # qid_pd = query_data_all[["qid"]].compute()

            # load query vectors
            # per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)


            label_path = get_label_dataset_name(ds_name, k,target_recall,s,efS)
            min_cns_df = pd.read_csv(label_path)
            min_cns_df.set_index("qid", inplace=True)  # 以 qid 作为索引
            y = query_data_all["qid"].map(lambda qid: min_cns_df.loc[qid, "min_efSearch"])
            y_all = np.minimum(y, efS)
            # ===== 标签分布统计 =====
            y_np = np.asarray(y_all)

            total = len(y_np)
            num_k = np.sum(y_np == k)
            num_cns = np.sum(y_np == efS)

            print(f"[Label Stats] k={k}, CNS={efS}, total={total}")
            print(f"  label == k   : {num_k} ({num_k / total:.4%})")
            print(f"  label == CNS : {num_cns} ({num_cns / total:.4%})")

            # 可选：中间区间
            mid = np.sum((y_np > k) & (y_np < efS))
            print(f"  k < label < CNS : {mid} ({mid / total:.4%})")

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

                    input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                                       "avg_dist", "variance","avg_hop","entry_query_dist_ratio","dist_distribution_entropy",
                                       "cosine_mean","cosine_variance","cosine_direction_entropy"]

                    X_train = data_all[input_features].values

                    model_file = f"/root/DARTH-main-data/predictor_models/raet-cns-regression_CNSFull/{ds_name}/k{k}/efS{efS}_s{train_num}_nestim{n_estimators}_{selected_features}_{target_recall}_CNSFull_quantile{quantile}_Noquery_Noske_Nodensty.txt"
                    model = lgb.Booster(model_file=model_file)
                    y_pred = model.predict(X_train)

                    mse = mean_squared_error(y_all, y_pred)
                    mae = mean_absolute_error(y_all, y_pred)
                    r2 = r2_score(y_all, y_pred)
                    # p99 = compute_P99(y_all, y_pred)
                    # p1 = compute_P1(y_all, y_pred)

                    results[k][s][li][selected_features] = {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        # "p99": p99,
                        # "p1": p1,
                    }
            print(results)
            with open(f"/root/DARTH-main-data/predictor_models/raet-cns-regression/{ds_name}/k{k}/训练数据量_{train_num}_的误差_{target_recall}.json", "w") as f:
                json.dump(results, f, indent=4)

    print(f"Finished predict for {ds_name}")
    print("\n")

    return results


if __name__ == "__main__":
    main()
