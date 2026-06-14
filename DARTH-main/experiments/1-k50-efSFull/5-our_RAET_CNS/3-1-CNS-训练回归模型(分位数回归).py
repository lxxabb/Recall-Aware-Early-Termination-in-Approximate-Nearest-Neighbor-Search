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
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
def get_label_dataset_name(ds_name, k,target_recall,num_queries,efS):
    recall_str = f"{target_recall:.2f}"
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/efs{efS}_min_ef_per_query_s{num_queries}_{recall_str}.csv"

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
    # quantile=0.72#sift
    # quantile=0.77#glove
    # quantile=0.78#gist
    quantile=0.72#deep
    model_conf = {
        "raet-cns-quantile": lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,                # glove CNS500，最优alpha是0.7
            random_state=SEED,
            n_estimators=n_estimators,
            verbose=-1,
        )
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
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
        },
    }

    training_queries_num = [10000]
    all_k_values = [50]
    # all_datasets = ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]
    all_datasets = ["SIFT1M"]
    # all_datasets = ["GLOVE100"]
    # all_datasets = ["GIST1M"]
    # all_datasets = ["DEEP10M"]
    all_li = [1]
    target_recalls = [0.90,0.92,0.94,0.96,0.98]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            M, efC, efS, d,n_estimators, columns_to_load, feature_classes, model_conf,target_recalls,quantile)

        with open(f"/root/DARTH-main-data/predictor_models/raet-cns-regression/k50_{ds_name}_qs10000_log.json", "w") as f:
            json.dump(results, f, indent=4)

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

def get_CNS(ds_name,target_recall):
    if ds_name == "SIFT1M":
        if target_recall==0.90:
            return 500
        if target_recall==0.92:
            return 500
        if target_recall==0.94:
            return 500
        if target_recall==0.96:
            return 500
        if target_recall==0.98:
            return 500
    if ds_name == "GLOVE100":
        if target_recall==0.90:
            return 500
        if target_recall==0.92:
            return 500
        if target_recall==0.94:
            return 500
        if target_recall==0.96:
            return 500
        if target_recall==0.98:
            return 500
    if ds_name == "GIST1M":
        if target_recall==0.90:
            return 1000
        if target_recall==0.92:
            return 1000
        if target_recall==0.94:
            return 1000
        if target_recall==0.96:
            return 1000
        if target_recall==0.98:
            return 1000
    if ds_name == "DEEP10M":
        if target_recall==0.90:
            return 750
        if target_recall==0.92:
            return 750
        if target_recall==0.94:
            return 750
        if target_recall==0.96:
            return 750
        if target_recall==0.98:
            return 750


def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls,quantile):
    results = {}
    quantile=int(quantile*100)
    max_query_size = max(training_queries_num)

    print(f"Starting training for {ds_name}")

    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, li)

        print(f"{ds_name} | k={k} | Path: {data_path}")

        # all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        all_queries_pd = pd.read_csv(data_path, usecols=columns_to_load)

        # all_queries_data = all_queries_dask.compute()


        for s in training_queries_num:
            for target_recall in target_recalls:
                CNS = get_CNS(ds_name,target_recall)
                if CNS==k:
                    #CNS=k的情况下去，不必训练模型
                    continue
                print(f"准确率为 {target_recall}")
                results[k][s] = {}
                query_data_all = all_queries_pd[all_queries_pd["qid"] < s]
                # query_data_all = all_queries_data[all_queries_data["qid"] < s]
                qid_pd = query_data_all[["qid"]]
                # Load the dimensions of the queries
                per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)

                #获取标签
                label_path = get_label_dataset_name(ds_name, k,target_recall,max_query_size,efS)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Min CNS file not found: {label_path}")
                min_cns_df = pd.read_csv(label_path)
                min_cns_df.set_index("qid", inplace=True)  # 以 qid 作为索引

                for li in all_li:
                    results[k][s][li] = {}

                    data_all = query_data_all[query_data_all["dists"] % li == 0]
                    # data_all = query_data_all[query_data_all["dists"] % li == 0]

                    print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")

                    y = data_all["qid"].map(lambda qid: min_cns_df.loc[qid, "min_efSearch"])

                    # 硬截断，最大值为CNS:训练集达到要求准确率的最小CNS
                    # y = np.minimum(y, CNS)
                    # a = k
                    # b = CNS
                    # eps = 1e-6
                    #
                    # z = (y - a) / (b - a)
                    # z = np.clip(z, eps, 1 - eps)
                    #
                    # y_all = np.log(z / (1 - z))

                    y_all = np.minimum(y, CNS)

                    # 2. log 变换
                    # y_all = np.log1p(y_all)

                    # ===== 标签分布统计 =====
                    y_np = np.asarray(y_all)

                    total = len(y_np)
                    num_k = np.sum(y_np == k)
                    num_cns = np.sum(y_np == CNS)

                    print(f"[Label Stats] k={k}, CNS={CNS}, total={total}")
                    print(f"  label == k   : {num_k} ({num_k / total:.4%})")
                    print(f"  label == CNS : {num_cns} ({num_cns / total:.4%})")

                    # 可选：中间区间
                    mid = np.sum((y_np > k) & (y_np < CNS))
                    print(f"  k < label < CNS : {mid} ({mid / total:.4%})")

                    for selected_features in feature_classes:
                        feats = feature_classes[selected_features]

                        data_df = data_all[feats]
                        per_query_dimensions_pd = per_query_dimensions_df.copy()
                        data_df = data_df.merge(per_query_dimensions_pd,left_on="qid",right_index=True)

                        query_dims_features = [f"d{d}" for d in dimensions]
                        # input_features =  ["dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                        #                    "avg_dist","avg_hop","entry_query_dist_ratio","dist_distribution_entropy","kurtosis",
                        #                    "cosine_mean"] + query_dims_features
                        input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                                           "avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy",
                                           "cosine_mean","cosine_variance","cosine_direction_entropy"] + query_dims_features

                        X_train = data_df[input_features].values
                        y_train = y_all

                        # model = model_conf["raet-cns"]
                        model = model_conf["raet-cns-quantile"]
                        model_train_time_start = time.time()

                        weights = np.ones_like(y_np, dtype=float)
                        # 强烈建议只对 CNS 加权
                        # weights[y_np == CNS] = num_k/num_cns+1
                        # p_cns = num_cns / total
                        # weights[y_np == CNS] = (1 - p_cns) / p_cns
                        # model.fit(X_train, y_train,sample_weight=weights)

                        model.fit(X_train, y_train)

                        model_train_time = time.time() - model_train_time_start

                        model_params = model.get_params()
                        learning_rate = model_params["learning_rate"]

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
                        recall_str = f"{target_recall:.2f}"
                        fi_file = f"/root/DARTH-main-data/feature_importance/raet-cns-regression_CNSFull/{ds_name}/k{k}/efS{CNS}_s{s}_nestim{n_estimators}_{selected_features}_{recall_str}_CNSFull_quantile{quantile}.csv"
                        # 保存到文件
                        os.makedirs(os.path.dirname(fi_file), exist_ok=True)
                        feature_importances.to_csv(fi_file, index=False)
                        print(f"特征重要性保存到: {fi_file}")

                        model_file = f"/root/DARTH-main-data/predictor_models/raet-cns-regression_CNSFull/{ds_name}/k{k}/efS{CNS}_s{s}_nestim{n_estimators}_{selected_features}_{recall_str}_CNSFull_quantile{quantile}.txt"
                        os.makedirs(os.path.dirname(model_file), exist_ok=True)
                        model.booster_.save_model(model_file)

                        print(f"模型保存到: {model_file}")
                        print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")

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