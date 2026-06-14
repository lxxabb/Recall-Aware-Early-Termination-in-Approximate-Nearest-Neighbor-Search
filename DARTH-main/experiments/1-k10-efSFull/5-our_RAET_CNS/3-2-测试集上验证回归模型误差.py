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
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/test/efS{efS}_qs{num_queries}.txt"
    # return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
def get_train_label_dataset_name(ds_name, k,target_recall,num_queries):
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/min_ef_per_query_s{num_queries}_{target_recall}.csv"
def get_test_label_dataset_name(ds_name, k,target_recall):
    return f"/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/test/min_ef_per_query_{target_recall}.csv"

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
    # train_num=10000
    train_num=100000
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
            label_path = get_test_label_dataset_name(ds_name, k, target_recall)
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


                    # y_pred_logit = model.predict(X_train)
                    # a = k
                    # b = CNS
                    # z = 1 / (1 + np.exp(-y_pred_logit))
                    # y_pred = a + z * (b - a)
                    # print("预测值最大 =", y_pred.max())
                    # y_pred = np.exp(y_pred_log)

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

                    # # ---- 分段误差 ----
                    # y = y_train
                    # p = y_pred
                    #
                    # # ① 100~400 区间
                    # mask_100_400 = (y >= 100) & (y <= 400)
                    #
                    # if mask_100_400.sum() > 0:
                    #     mse_100_400 = mean_squared_error(y[mask_100_400], p[mask_100_400])
                    #     mae_100_400 = mean_absolute_error(y[mask_100_400], p[mask_100_400])
                    #     p99_100_400 = compute_P99(y[mask_100_400], p[mask_100_400])
                    #     p1_100_400  = compute_P1(y[mask_100_400], p[mask_100_400])
                    # else:
                    #     mse_100_400 = mae_100_400 = p99_100_400 = p1_100_400 = None
                    #
                    #
                    # # ② 400 以上
                    # mask_400_plus = (y > 400)
                    #
                    # if mask_400_plus.sum() > 0:
                    #     mse_400_plus = mean_squared_error(y[mask_400_plus], p[mask_400_plus])
                    #     mae_400_plus = mean_absolute_error(y[mask_400_plus], p[mask_400_plus])
                    #     p99_400_plus = compute_P99(y[mask_400_plus], p[mask_400_plus])
                    #     p1_400_plus  = compute_P1(y[mask_400_plus], p[mask_400_plus])
                    # else:
                    #     mse_400_plus = mae_400_plus = p99_400_plus = p1_400_plus = None
                    #
                    # results[k][s][li][selected_features] = {
                    #     "mse": mse,
                    #     "mae": mae,
                    #     "r2": r2,
                    #     "p99": p99,
                    #     "p1": p1,
                    #
                    #     "mse_100_400": mse_100_400,
                    #     "mae_100_400": mae_100_400,
                    #     "p99_100_400": p99_100_400,
                    #     "p1_100_400": p1_100_400,
                    #
                    #     "mse_400_plus": mse_400_plus,
                    #     "mae_400_plus": mae_400_plus,
                    #     "p99_400_plus": p99_400_plus,
                    #     "p1_400_plus": p1_400_plus,
                    # }



            print(results)

    print(f"Finished predict for {ds_name}")
    print("\n")

    return results


if __name__ == "__main__":
    main()
#1w训练数据 未删除特征截断  测试集合效果
# Starting training for SIFT1M
#     SIFT1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/test/efS500_qs10000.txt
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 6.104906123361346, 'mae': 1.7586926866831352, 'r2': 0.40556138857883495, 'p99': 6.5191094339968245, 'p1': 0.0}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 85.9809588840051, 'mae': 6.7684946617321575, 'r2': 0.44749891613752724, 'p99': 24.793779241008604, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 485.8906246766597, 'mae': 16.50273402066315, 'r2': 0.4539975596948258, 'p99': 59.62099130339686, 'p1': 0.0}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/test/efS1000_qs1000.txt
# 准确率为 0.95
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 2219.3670662593645, 'mae': 36.42138219509985, 'r2': 0.5483001960011065, 'p99': 122.49313267869543, 'p1': 0.0}}}}}
# 准确率为 0.96
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 3392.1755460384443, 'mae': 44.98075940970119, 'r2': 0.5528062942985961, 'p99': 150.49572802848454, 'p1': 0.0}}}}}
# 准确率为 0.97
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 5808.635799049674, 'mae': 59.82627397576677, 'r2': 0.5360564331762874, 'p99': 196.3143332501581, 'p1': 0.0}}}}}
# 准确率为 0.98
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 10982.550426542517, 'mae': 82.09107421556646, 'r2': 0.5278585693134825, 'p99': 253.70160792132305, 'p1': 0.0}}}}}
# 准确率为 0.99
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 32416.417135755248, 'mae': 143.46790956601146, 'r2': 0.47990018995226424, 'p99': 434.04985637075157, 'p1': 0.0}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/test/efS500_qs10000.txt
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 17.438183723548406, 'mae': 2.34793647370082, 'r2': 0.5053217848596478, 'p99': 14.570807567968027, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 863.9326582888914, 'mae': 16.11487603380444, 'r2': 0.5982421918274208, 'p99': 116.00736234919007, 'p1': 0.0}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/test/efS750_qs10000.txt
# 准确率为 0.96
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 26.60033801617357, 'mae': 3.2767553835904772, 'r2': 0.5705572307065389, 'p99': 15.555200327155129, 'p1': 0.0}}}}}
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 136.42631122698583, 'mae': 7.615224065148754, 'r2': 0.6009758591781272, 'p99': 35.20240493254338, 'p1': 0.0}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 466.5343632309776, 'mae': 14.484201824538852, 'r2': 0.6224292881562092, 'p99': 65.8980192796277, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 1763.1457574223482, 'mae': 29.26005361482959, 'r2': 0.6264827745165367, 'p99': 122.53727809412412, 'p1': 0.0}}}}}
# Finished predict for DEEP10M


#10w 训练数据  测试集合效果
# SIFT1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/test/efS500_qs10000.txt
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 5.76911609318421, 'mae': 1.682539999757507, 'r2': 0.438257478450511, 'p99': 6.383611308722136, 'p1': 0.0}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 81.66405720820373, 'mae': 6.502256803493024, 'r2': 0.4752386958022977, 'p99': 24.20834318430321, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 458.74318132159544, 'mae': 15.915675067759134, 'r2': 0.48450354101474824, 'p99': 58.02477095822243, 'p1': 0.0}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/test/efS1000_qs1000.txt
# 准确率为 0.95
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 1893.100232594377, 'mae': 33.32279797845033, 'r2': 0.6147041122609829, 'p99': 119.2983364615848, 'p1': 0.0}}}}}
# 准确率为 0.96
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 2892.5027974344102, 'mae': 41.17001193977057, 'r2': 0.6186786246227742, 'p99': 137.66572609281118, 'p1': 0.0}}}}}
# 准确率为 0.97
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 4897.86058387398, 'mae': 54.62115717054346, 'r2': 0.6088012766337437, 'p99': 184.20055425021414, 'p1': 0.0}}}}}
# 准确率为 0.98
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 9490.62761689842, 'mae': 76.0331670817889, 'r2': 0.591996546601238, 'p99': 243.6677950254898, 'p1': 0.0}}}}}
# 准确率为 0.99
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'mse': 28223.922908036668, 'mae': 132.32956353436796, 'r2': 0.5471659658808927, 'p99': 420.1389949097931, 'p1': 0.0}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/test/efS500_qs10000.txt
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 16.326674085855945, 'mae': 2.259365484249857, 'r2': 0.5368525688221143, 'p99': 14.35339724863406, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 790.4645552573578, 'mae': 15.435731410114846, 'r2': 0.6324073362531523, 'p99': 109.8470385954004, 'p1': 0.0}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/test/efS750_qs10000.txt
# 准确率为 0.96
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 24.593792979360167, 'mae': 3.1140095611060237, 'r2': 0.6029514152013851, 'p99': 15.028166302552219, 'p1': 0.0}}}}}
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 126.6456084017336, 'mae': 7.2276431359926105, 'r2': 0.6295827789605362, 'p99': 34.70499577120139, 'p1': 0.0}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 427.17934185879994, 'mae': 13.707824797011122, 'r2': 0.6542796824791757, 'p99': 64.05973283241063, 'p1': 0.0}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'mse': 1608.8857369927484, 'mae': 27.790453202501038, 'r2': 0.6591623046071873, 'p99': 119.01574691259054, 'p1': 0.0}}}}}
# Finished predict for DEEP10M
