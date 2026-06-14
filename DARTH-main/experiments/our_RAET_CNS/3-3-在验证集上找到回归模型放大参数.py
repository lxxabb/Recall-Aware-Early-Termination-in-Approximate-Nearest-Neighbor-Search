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
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/validation/efS{efS}_qs{num_queries}.txt"
    # return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
def get_validation_label_dataset_name(ds_name, k,target_recall,query_num):
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/validation/min_ef_per_query_s{query_num}_{target_recall}.csv"

def main():
    SEED = 42
    n_estimators = 100
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,visited_points,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
    qid_feat=["qid"]
    #索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
    index_metric_feats = ["step", "dists", "inserts"]
    #近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    #近邻统计特征，包括（近邻距离的平均值，方差，密度特征（平均距离的倒数）,入口点到查询点距离 与入口点距离均值之间的比值特征,距离分布熵 ,）
    neighbor_stats_feats = ["avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy"]
    #余弦特征：
    cosine_feats = ["cosine_mean","cosine_variance","cosine_direction_entropy"]


    all_feats = qid_feat+ index_metric_feats + neighbor_distances_feats + neighbor_stats_feats + cosine_feats

    columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

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

    test_queries_num = [10000]
    all_k_values = [100]
    # all_datasets = ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]
    all_datasets = ["GLOVE100"]
    all_li = [1]
    # target_recalls = [0.95,0.96,0.97,0.98,0.99]
    target_recalls = [0.98,0.99]

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

    # ---- 先读 query embedding ---- 需要改为验证集
    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/SIFT1M/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/GIST1M/gist_learn.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs(f"/root/ann-data/GLOVE100/validation.10K.fvecs", limit=s)
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

def get_CNS(ds_name,target_recall):
    if ds_name == "SIFT1M":
        if target_recall==0.99:
            return 172
        if target_recall==0.98:
            return 128
        if target_recall==0.97:
            return 107
        if target_recall==0.96:
            return 100
        if target_recall==0.95:
            return 100
    if ds_name == "GLOVE100":
        if target_recall==0.99:
            return 247
        if target_recall==0.98:
            return 116
        if target_recall==0.97:
            return 100
        if target_recall==0.96:
            return 100
        if target_recall==0.95:
            return 100
    if ds_name == "GIST1M":
        if target_recall==0.99:
            return 788
        if target_recall==0.98:
            return 503
        if target_recall==0.97:
            return 385
        if target_recall==0.96:
            return 316
        if target_recall==0.95:
            return 270
    if ds_name == "DEEP10M":
        if target_recall==0.99:
            return 271
        if target_recall==0.98:
            return 181
        if target_recall==0.97:
            return 141
        if target_recall==0.96:
            return 117
        if target_recall==0.95:
            return 100


def train_predictors(ds_name, test_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls):
    results = {}
    #使用多大训练数据，训练的模型，有1w和10w
    train_num=10000
    # train_num=100000
    print(f"Starting training for {ds_name}")
    s=test_queries_num
    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, s, ds_name, k, li)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        query_data_all = dd.read_csv(data_path, usecols=columns_to_load)
        # all_queries_data = all_queries_dask.compute()
        for target_recall in target_recalls:
            CNS = get_CNS(ds_name,target_recall)
            if CNS==k:
                continue
            print(f"准确率为 {target_recall}")
            results[k][s] = {}

            # compute qid df
            qid_pd = query_data_all[["qid"]].compute()

            # load query vectors
            per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)

            # --- load labels ---
            label_path = get_validation_label_dataset_name(ds_name, k, target_recall,s)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Min CNS file not found: {label_path}")

            min_cns_df = pd.read_csv(label_path).set_index("qid")

            for li in all_li:
                results[k][s][li] = {}

                # filter + compute (IMPORTANT FIX)
                data_all = query_data_all[query_data_all["dists"] % li == 0].compute()

                print(f"    {s} Queries Data Shape: {data_all.shape} | Li: {li}")

                # merge label (IMPORTANT FIX)
                data_all = data_all.merge(
                    min_cns_df[["min_efSearch"]],
                    left_on="qid",
                    right_index=True,
                    how="inner"
                )

                assert len(data_all) > 0, "data_all empty after merge — check labels/qid alignment"

                for selected_features in feature_classes:
                    feats = feature_classes[selected_features]

                    data_df = data_all[feats].merge(
                        per_query_dimensions_df,
                        left_on="qid",
                        right_index=True,
                        how="left"
                    )

                    query_dims_features = [f"d{i}" for i in dimensions]
                    # input_features = feats[1:] + query_dims_features  # skip qid
                    # X_train = data_df[input_features].values

                    input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                                       "avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy",
                                       "cosine_mean","cosine_variance","cosine_direction_entropy"] + query_dims_features

                    # input_features =  ["dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                    #                    "avg_dist","avg_hop","entry_query_dist_ratio","dist_distribution_entropy","kurtosis",
                    #                    "cosine_mean"] + query_dims_features
                    X_train = data_df[input_features].values


                    y_train = data_all["min_efSearch"].values
                    # 硬截断，最大值为efS
                    y_train = np.minimum(y_train, CNS)

                    # Load trained model
                    model_file = f"/root/DARTH-main-data/predictor_models/raet-cns-regression/{ds_name}/k{k}/efS{efS}_s{train_num}_nestim{n_estimators}_{selected_features}_{target_recall}_JieDuan.txt"
                    model = lgb.Booster(model_file=model_file)
                    y_pred = model.predict(X_train)
                    y_pred[y_pred < k] = k
                    y_pred[y_pred > CNS] = CNS

                    #-----使用分位数算γ-----
                    # eps = 1e-6  # 防止除零
                    # ratio = y_train / (y_pred + eps)
                    # if target_recall >= 0.99:
                    #     quantile_p = 0.95
                    # elif target_recall >= 0.98:
                    #     quantile_p = 0.90
                    # #Glove数据集0.98,参数值为 1.0328
                    # #Glove数据集0.99,参数值为 1.3171
                    # gamma = np.quantile(ratio, quantile_p)

                    #-----使用最小二乘法算γ-----gamma值小于1，不合适

                    y_label = y_train.astype(np.float64)
                    # ef_pred
                    y_hat = y_pred.astype(np.float64)
                    # 最小二乘 gamma（解析解）
                    numerator = np.sum(y_hat * y_label)
                    denominator = np.sum(y_hat * y_hat) + 1e-12  # 防止除 0
                    gamma = numerator / denominator

                    print(f"gamma参数值为 {gamma}")
                    ef_final = np.ceil(gamma * y_pred).astype(int)

                    # safety floor & ceiling（强烈建议）
                    ef_final = np.maximum(ef_final, k)
                    ef_final = np.minimum(ef_final, CNS)

                    coverage = np.mean(ef_final >= y_train)
                    print(f"放大后的CNS满足要求的占比，coverage = {coverage:.4f}")

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

#GLOVE 数据集0.98 放大参数为1.0328
#GLOVE 数据集0.99 放大参数为1.3171