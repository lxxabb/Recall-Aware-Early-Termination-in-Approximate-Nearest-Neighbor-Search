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
    #glove
    # return f"/root/DARTH-main-data/data/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}_visitedPoints_filtered_group_cap.txt"
    #sift
    return f"/root/DARTH-main-data/data/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}_visitedPoints_filtered_r0.9.txt"

def main():
    SEED = 42
    n_estimators = 100
    #qid,step,dists,elaps_ms,inserts,first_nn_dist,visited_points,nn_dist,avg_dist,furthest_dist,
    # percentile_25,percentile_50,percentile_75,percentile_95,
    # variance,std,range,skewness,kurtosis,energy,nn10_dist,nn_to_first,nn10_to_first,RDE,TDR,NRS,feats_collect_time_ms,r
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
        "raet": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
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
            "d":128
        },
        # "SIFT100M": {
        #     "M": 32,
        #     "efC": 500,
        #     "efS": 500,
        # },
        # "GIST1M": {
        #     "M": 32,
        #     "efC": 500,
        #     "efS": 1000,
        # },
        # "GLOVE100": {
        #     "M": 16,
        #     "efC": 500,
        #     "efS": 500,
        #     "d": 100
        # },
        # "DEEP100M": {
        #     "M": 32,
        #     "efC": 500,
        #     "efS": 750,
        # },
        # "T2I100M": {
        #     "M": 80,
        #     "efC": 1000,
        #     "efS": 2500,
        # }
    }

    training_queries_num = [10000]
    all_k_values = [100]
    all_datasets = ["SIFT1M"]
    all_li = [1]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            M, efC, efS, d,n_estimators, columns_to_load, feature_classes, model_conf)

        # with open(f"./training_results_{ds_name}_u10K.json", "w") as f:
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
def get_query_dims_df(qid_df, d, ds_name, s):
    """
    qid_df：必须是 pandas DataFrame，只包含 qid 列
    """

    # ---- 先读 query embedding ----
    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs("/root/ann-data/sift/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs("/data/mchatzakis/datasets/processed/GIST1M/learn.100K.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs("/root/ann-data/glove-100/glove原本数据集/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP100M":
        query_dims, dims_read = read_fvecs("/data/mchatzakis/datasets/processed/DEEP100M/learn.1M.fvecs", limit=s)
    elif ds_name == "T2I100M":
        query_dims, dims_read = read_fvecs("/data/mchatzakis/datasets/processed/T2I100M/learn.1M.fvecs", limit=s)

    assert query_dims.shape[0] == s
    assert query_dims.shape[1] == d
    assert dims_read == d

    # ---- 取 qid（pandas，速度快）----
    qids = qid_df["qid"].unique()
    qids.sort()

    # ---- 直接构建 DataFrame（一次完成，不卡顿）----
    dimensions = np.arange(1, d + 1)
    col_names = [f"d{dim}" for dim in dimensions]

    per_query_dimensions_df = pd.DataFrame(
        data=query_dims,
        index=qids,
        columns=col_names
    )

    return per_query_dimensions_df, dimensions


def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf):
    results = {}

    max_query_size = max(training_queries_num)

    print(f"Starting training for {ds_name}")

    for k in all_k_values:
        results[k] = {}
        li = 2

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, li)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        # all_queries_data = all_queries_dask.compute()



        for s in training_queries_num:
            results[k][s] = {}
            query_data_all = all_queries_dask[all_queries_dask["qid"] < s]
            # query_data_all = all_queries_data[all_queries_data["qid"] < s]
            qid_pd = query_data_all[["qid"]].compute()
            # Load the dimensions of the queries
            per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)

            for li in all_li:
                results[k][s][li] = {}

                data_all = query_data_all[query_data_all["dists"] % li == 0]
                # data_all = query_data_all[query_data_all["dists"] % li == 0]

                print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")

                y_all = data_all["r"]

                for selected_features in feature_classes:
                    feats = feature_classes[selected_features]

                    data_df = data_all[feats]
                    per_query_dimensions_pd = per_query_dimensions_df.copy()
                    data_df = data_df.merge(
                        per_query_dimensions_pd,
                        left_on="qid",
                        right_index=True
                    )
                    data_df = data_df.compute()
                    # data_df = data_df.merge(per_query_dimensions_df, left_on="qid", right_index=True)


                    query_dims_features = [f"d{d}" for d in dimensions]
                    # input_features =  ["step", "dists", "inserts","visited_points","first_nn_dist", "nn_dist", "furthest_dist","avg_dist", "variance"] + query_dims_features
                    #删除入口点距离first_nn_dist
                    input_features =  ["step", "dists", "visited_points", "nn_dist", "furthest_dist","avg_dist", "variance"] + query_dims_features
                    X_train = data_df[input_features].values
                    y_train = y_all

                    model = model_conf["raet"]
                    model_train_time_start = time.time()
                    model.fit(X_train, y_train)
                    model_train_time = time.time() - model_train_time_start

                    model_params = model.get_params()
                    learning_rate = model_params["learning_rate"]

                    # feature_importances = pd.DataFrame({'Feature': input_features, 'Importance': model.feature_importances_})
                    # feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

                    # 生成特征重要性 DataFrame
                    feature_importances = pd.DataFrame({
                        'Feature': input_features,
                        'Importance': model.feature_importances_
                    })

                    # 计算占比（百分比）
                    total_importance = feature_importances['Importance'].sum()
                    feature_importances['Importance_Ratio'] = feature_importances['Importance'] / total_importance
                    # 排序
                    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

                    # 构造保存路径（和模型同名更清晰）
                    fi_file = f"/root/DARTH-main-data/feature_importance/{ds_name}/raet_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}_feature_importance_删除入口点距离和插入前k位置数量.csv"

                    # 保存到文件
                    feature_importances.to_csv(fi_file, index=False)

                    print(f"Feature importance saved to: {fi_file}")


                    model_file = f"/root/DARTH-main-data/predictor_models/raet/{ds_name}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}_filtered_r0.9_删除入口点距离和插入前k位置数量.txt"
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