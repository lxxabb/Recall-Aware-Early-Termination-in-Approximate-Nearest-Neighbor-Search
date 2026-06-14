import os

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
def get_label_dataset_name(ds_name, k,target_recall):
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
        "raet-cns": lgb.LGBMRegressor(objective='binary', random_state=SEED, n_estimators=n_estimators, verbose = -1),
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

    test_query_num = [100000]
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

        # with open(f"/root/DARTH-main-data/predictor_models/raet-cns-classification/{ds_name}_qs10000.json", "w") as f:
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
def train_predictors(ds_name, test_query_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls):
    results = {}
    train_num=100000
    print(f"Starting training for {ds_name}")
    s=test_query_num
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
            label_path = get_label_dataset_name(ds_name, k, target_recall)
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
                    # X_test = data_df[input_features].values
                    input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                                       "avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy",
                                       "cosine_mean","cosine_variance","cosine_direction_entropy"] + query_dims_features
                    # input_features =  ["dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                    #                    "avg_dist","avg_hop","entry_query_dist_ratio","dist_distribution_entropy","kurtosis",
                    #                    "cosine_mean"] + query_dims_features
                    X_test = data_df[input_features].values
                    y_test = data_all["min_efSearch"].values
                    # --- 生成二分类标签 ---
                    y_test = (y_test > k).astype(int)

                    # Load trained model
                    model_file = f"/root/DARTH-main-data/predictor_models/raet-cns-classification/{ds_name}/k{k}/efS{efS}_s{train_num}_nestim{n_estimators}_{selected_features}_{target_recall}.txt"
                    model = lgb.Booster(model_file=model_file)
                    y_prob = model.predict(X_test)     # probability of class 1
                    y_cls = (y_prob >= 0.5).astype(int)
                    # y_pred = np.exp(y_pred_log)
                    acc = accuracy_score(y_test, y_cls)
                    f1 = f1_score(y_test, y_cls)
                    auc = roc_auc_score(y_test, y_prob)

                    results[k][s][li][selected_features] = {
                        "acc": acc,
                        "f1": f1,
                        "auc": auc,
                    }

                    print(results)

    print(f"Finished predict for {ds_name}")
    print("\n")

    return results


if __name__ == "__main__":
    main()
#二分类，删除特征
# Starting training for SIFT1M
#     SIFT1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/test/efS500_qs10000.txt
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8557, 'f1': 0.5861772297103527, 'auc': 0.8892422870563503}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8215, 'f1': 0.6223820605034905, 'auc': 0.8788031460016518}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.798, 'f1': 0.6903740036787247, 'auc': 0.8704315728123005}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.7863, 'f1': 0.7582852618482072, 'auc': 0.8726713601251985}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.7862, 'f1': 0.8201850294365013, 'auc': 0.8676541454271239}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/test/efS1000_qs1000.txt
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.887, 'f1': 0.931139549055454, 'auc': 0.9331421401684888}}}}}
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.907, 'f1': 0.9453904873752202, 'auc': 0.9422643399849712}}}}}
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.917, 'f1': 0.9529745042492918, 'auc': 0.9508334778193348}}}}}
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.931, 'f1': 0.9623978201634877, 'auc': 0.9573822825219472}}}}}
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.945, 'f1': 0.97094558901215, 'auc': 0.9518745266346881}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/test/efS500_qs10000.txt
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.9432, 'f1': 0.6316472114137484, 'auc': 0.9619098086897744}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.9305, 'f1': 0.6393357550596782, 'auc': 0.9517351389535066}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.9123, 'f1': 0.6515693285657529, 'auc': 0.9280926199194941}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8721, 'f1': 0.6597499334929503, 'auc': 0.8985759667692252}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.7948, 'f1': 0.6664499349804941, 'auc': 0.8404907302512092}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/test/efS750_qs10000.txt
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.846, 'f1': 0.7580138277812697, 'auc': 0.9243230746724498}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8485, 'f1': 0.7979730630750766, 'auc': 0.9260368375967039}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8422, 'f1': 0.8214528173794976, 'auc': 0.9236630810661468}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8412, 'f1': 0.8509480007508917, 'auc': 0.9265705270607087}}}}}
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8439, 'f1': 0.8769024524879742, 'auc': 0.9221915048788529}}}}}
# Finished predict for DEEP10M


#二分类 全部特征 10w训练集
# SIFT1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/test/efS500_qs10000.txt
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8152, 'f1': 0.720508166969147, 'auc': 0.8886691571147066}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.7961, 'f1': 0.7711303176563026, 'auc': 0.8873644085610699}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8027, 'f1': 0.8335442504007424, 'auc': 0.8835017237747023}}}}}
# Finished predict for SIFT1M
#
#
# Starting training for GIST1M
#     GIST1M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GIST1M/k100/test/efS1000_qs1000.txt
# 准确率为 0.95
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.894, 'f1': 0.9352869352869353, 'auc': 0.9458189886952921}}}}}
# 准确率为 0.96
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.9, 'f1': 0.9407582938388626, 'auc': 0.9474826751273273}}}}}
# 准确率为 0.97
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.905, 'f1': 0.9457452884066248, 'auc': 0.949958305468176}}}}}
# 准确率为 0.98
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.923, 'f1': 0.9577155409115871, 'auc': 0.9529130087789306}}}}}
# 准确率为 0.99
# 1000 Queries Data Shape: (1000, 22) | Li: 1
# {100: {1000: {1: {'all_feats': {'acc': 0.935, 'f1': 0.9654806160382369, 'auc': 0.9505333249179501}}}}}
# Finished predict for GIST1M
#
#
# Starting training for GLOVE100
#     GLOVE100 | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/GLOVE100/k100/test/efS500_qs10000.txt
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8802, 'f1': 0.6739248775176919, 'auc': 0.9075301578067931}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8053, 'f1': 0.6754459076512752, 'auc': 0.8589013864163565}}}}}
# Finished predict for GLOVE100
#
#
# Starting training for DEEP10M
#     DEEP10M | k=100 | Path: /root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/DEEP10M/k100/test/efS750_qs10000.txt
# 准确率为 0.96
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8604, 'f1': 0.8148541114058355, 'auc': 0.9360496334437871}}}}}
# 准确率为 0.97
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8535, 'f1': 0.8351153629713, 'auc': 0.9338865865346176}}}}}
# 准确率为 0.98
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8508, 'f1': 0.8601162572660791, 'auc': 0.934373289235707}}}}}
# 准确率为 0.99
# 10000 Queries Data Shape: (10000, 22) | Li: 1
# {100: {10000: {1: {'all_feats': {'acc': 0.8548, 'f1': 0.8863849765258216, 'auc': 0.9298931712378309}}}}}
# Finished predict for DEEP10M