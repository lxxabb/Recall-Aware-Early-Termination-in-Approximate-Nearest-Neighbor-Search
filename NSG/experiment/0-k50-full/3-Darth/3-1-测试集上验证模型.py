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
    return f"/root/NSG-data/test_data/Laet_Darth_test_data/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"

def main():
    SEED = 42
    n_estimators = 100
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,visited_points,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
    #索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
    index_metric_feats = ["step", "dists", "inserts"]
    #近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    #近邻统计特征，包括（近邻距离的平均值，方差，25，50，75分位数）
    neighbor_stats_feats = ["mean_distance", "variance_of_distances", "percentile_25", "percentile_50", "percentile_75"]
    all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

    columns_to_load = ["qid"] + all_feats + ["r"]

    model_conf = {
        "darth": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
    }

    feature_classes = {
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

def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls):
    results = {}
    train_num=10000
    print(f"Starting training for {ds_name}")
    s=training_queries_num
    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, s, ds_name, k, li)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        query_data_all = dd.read_csv(data_path, usecols=columns_to_load,dtype={'r': 'float64'})
        # all_queries_data = all_queries_dask.compute()

        results[k][s] = {}

        y_train = query_data_all["r"].compute()

        for li in all_li:
            results[k][s][li] = {}

            for selected_features in feature_classes:
                feats = feature_classes[selected_features]

                input_features =  ["step", "dists", "inserts","first_nn_dist","nn_dist","furthest_dist","mean_distance","variance_of_distances","percentile_25","percentile_50","percentile_75"]
                X_train = query_data_all[input_features].values

                model_file = f"/root/NSG-data/predictor_models/darth/{ds_name}/k{k}/efS{efS}_s10000_nestim{n_estimators}_{selected_features}.txt"
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
# SIFT1M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_test_data/SIFT1M/k100/efS500_qs10000.txt
# {100: {10000: {1: {'all_feats': {'mse': 0.005541783032506413, 'mae': 0.056340642515326014, 'r2': 0.938987755478594, 'p99': 0.21225611909355024, 'p1': 0.000704793297193449}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_test_data/GIST1M/k100/efS1000_qs1000.txt
# {100: {1000: {1: {'all_feats': {'mse': 0.011378713814251455, 'mae': 0.07920442862689288, 'r2': 0.8702743947451551, 'p99': 0.31403348832311717, 'p1': 0.0009631559160065833}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_test_data/GLOVE100/k100/efS500_qs10000.txt
# {100: {10000: {1: {'all_feats': {'mse': 0.0031082271614888825, 'mae': 0.038447445644285504, 'r2': 0.9653370086383559, 'p99': 0.1862621036310108, 'p1': 0.0004940013529458609}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/NSG-data/test_data/Laet_Darth_test_data/DEEP10M/k100/efS750_qs10000.txt
# {100: {10000: {1: {'all_feats': {'mse': 0.007382407594193459, 'mae': 0.06379319365054574, 'r2': 0.915164802268192, 'p99': 0.25222739832193647, 'p1': 0.0008169002829214623}}}}}
# Finished predict for DEEP10M
