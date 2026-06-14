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


def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"/root/DARTH-main-data/data/et_training_data/early-stop-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"

def main():
    SEED = 42
    n_estimators = 100
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
    #索引特征，包括（搜索步数，距离计算次数，有多少个数据点插入到CNS前K位置）
    index_metric_feats = ["step", "dists", "inserts"]
    #近邻距离特征，包括（最底层入口点距离，当前最近邻距离CNS中0号位置，CNS中K号位置）
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    #近邻统计特征，包括（近邻距离的平均值，方差，25，50，75分位数）
    neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
    all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

    columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

    model_conf = {
        "darth-our": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
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
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
        },
    }

    training_queries_num = [10000]
    all_k_values = [100]
    all_datasets = ["GLOVE100"]
    all_li = [1]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]

        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf)

        with open(f"/root/DARTH-main-data/predictor_models/darth-our/{ds_name}_qs10000.json", "w") as f:
            json.dump(results, f, indent=4)

def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf):
    results = {}

    max_query_size = max(training_queries_num)

    print(f"Starting training for {ds_name}")

    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, 2)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        all_queries_data = all_queries_dask.compute()

        for s in training_queries_num:
            results[k][s] = {}
            query_data_all = all_queries_data[all_queries_data["qid"] < s]

            for li in all_li:
                results[k][s][li] = {}

                data_all = query_data_all[query_data_all["dists"] % li == 0]

                print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")

                y_all = data_all["r"]

                for selected_features in feature_classes:
                    feats = feature_classes[selected_features]

                    X_all = data_all[feats]
                    X_train, y_train = X_all, y_all

                    model = model_conf["darth-our"]
                    model_train_time_start = time.time()
                    model.fit(X_train, y_train)
                    model_train_time = time.time() - model_train_time_start

                    model_params = model.get_params()
                    learning_rate = model_params["learning_rate"]

                    feature_importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

                    model_file = f"/root/DARTH-main-data/predictor_models/darth-our/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}.txt"
                    os.makedirs(os.path.dirname(model_file), exist_ok=True)
                    model.booster_.save_model(model_file)
                    print(f"Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")

                    results[k][s][li][selected_features] = {
                        "training_time": model_train_time,
                        "training_data_size": X_train.shape[0],
                        "learning_rate": learning_rate,
                        "feature_importances": feature_importances.to_dict(orient="records"),
                        "n_estimators": n_estimators,
                    }

                    print(f"results[{k}][{s}][{li}][{selected_features}]:\n{results[k][s][li][selected_features]}")

            print()
        print("\n\n")
    print(f"Finished training for {ds_name}")

    return results


if __name__ == "__main__":
    main()