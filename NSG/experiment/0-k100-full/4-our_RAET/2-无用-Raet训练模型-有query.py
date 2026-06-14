import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
import dask.dataframe as dd
import dask
import gc


def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    # return f"/root/NSG-data/train_data/raet/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
    return f"/root/NSG-data/train_data/Laet_Darth_train_data/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"


def read_fvecs(file_path, limit=None):
    with open(file_path, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        vector_size = (dim + 1) * 4
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        num_vectors = file_size // vector_size
        if limit is not None and limit < num_vectors:
            num_vectors = limit

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)
        return data[:, 1:], int(dim)


def get_query_dims_df(d, ds_name, s):
    if ds_name == "SIFT1M":
        query_dims, _ = read_fvecs("/root/ann-data/SIFT1M/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, _ = read_fvecs("/root/ann-data/GIST1M/gist_learn.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, _ = read_fvecs("/root/ann-data/GLOVE100/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP10M":
        query_dims, _ = read_fvecs("/root/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs", limit=s)

    assert query_dims.shape == (s, d)
    return query_dims


def train_predictors(ds_name, training_queries_num, all_k_values, all_li,
                     M, efC, efS, d, n_estimators,
                     columns_to_load, feature_classes, model_conf):

    results = {}
    max_query_size = max(training_queries_num)

    print(f"Starting training for {ds_name}")

    for k in all_k_values:
        results[k] = {}

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, 1)
        print(f"{ds_name} | k={k} | Path: {data_path}")

        all_queries_dask = dd.read_csv(
            data_path,
            usecols=columns_to_load,
            blocksize="128MB",
            dtype={'r': 'float64', 'mean_distance': 'float64','nn_dist': 'float64'}
        )

        for s in training_queries_num:
            results[k][s] = {}

            query_data_all = all_queries_dask[all_queries_dask["qid"] < s]

            # 仅计算 qid，用完立刻释放
            # qid_pd = query_data_all[["qid"]].drop_duplicates().compute()
            # s_real = len(qid_pd)
            # del qid_pd
            # gc.collect()

            qid_pd = query_data_all[["qid"]].drop_duplicates().compute()
            max_qid = int(qid_pd["qid"].max())
            del qid_pd
            gc.collect()

            for li in all_li:
                results[k][s][li] = {}

                # 注意：query_dim_mat 不跨 li 生命周期
                # query_dim_mat = get_query_dims_df(d, ds_name, s_real)
                query_dim_mat = get_query_dims_df(d, ds_name, max_qid + 1)

                data_all = query_data_all[query_data_all["dists"] % li == 0]

                for selected_features in feature_classes:
                    feats = feature_classes[selected_features]

                    # 控制 Dask compute 的生命周期
                    with dask.config.set(scheduler="single-threaded"):
                        data_df = data_all[feats].compute()
                        y_all = data_all["r"].compute()

                    # 构造训练数据
                    qid_index = data_df["qid"].values
                    query_embed = query_dim_mat[qid_index]
                    dimensions = np.arange(1, d + 1)
                    query_dims_features = [f"d{d}" for d in dimensions]
                    data_df["visited_points"] = data_df["visited_points"] / k
                    print(data_df["visited_points"])
                    input_features=[
                        "step", "dists", "inserts",
                        "visited_points",
                        "duration",
                        "nn_dist",
                        "furthest_dist",
                        "mean_distance",
                        "variance_of_distances"
                    ]+ query_dims_features
                    X_core = data_df[
                        [
                            "step", "dists", "inserts",
                            "visited_points",
                            "duration",
                            "nn_dist",
                            "furthest_dist",
                            "mean_distance",
                            "variance_of_distances"
                        ]
                    ].values

                    X_train = np.hstack([X_core, query_embed])

                    # 中间变量立即释放
                    del X_core
                    del query_embed
                    del data_df
                    gc.collect()

                    model = model_conf["raet"]

                    t0 = time.time()
                    model.fit(X_train, y_all)
                    train_time = time.time() - t0

                    # fit 完立刻释放大对象
                    del X_train
                    del y_all
                    gc.collect()

                    model_params = model.get_params()
                    learning_rate = model_params["learning_rate"]

                    feature_importances = pd.DataFrame({
                        "Feature": input_features,
                        "Importance": model.feature_importances_
                    })
                    # 计算占比（百分比）
                    total_importance = feature_importances['Importance'].sum()
                    feature_importances['Importance_Ratio'] = feature_importances['Importance'] / total_importance
                    # 排序
                    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

                    # 构造保存路径（和模型同名更清晰）
                    fi_file = f"/root/NSG-data/feature_importance/raet/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}.csv"
                    # 保存到文件
                    os.makedirs(os.path.dirname(fi_file), exist_ok=True)
                    feature_importances.to_csv(fi_file, index=False)
                    print(f"特征重要性保存到: {fi_file}")

                    model_file = (
                        f"/root/NSG-data/predictor_models/raet/"
                        f"{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}.txt"
                    )
                    os.makedirs(os.path.dirname(model_file), exist_ok=True)
                    model.booster_.save_model(model_file)

                    results[k][s][li][selected_features] = {
                        "training_time": train_time,
                        "training_data_size": int(feature_importances.shape[0]),
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                    }

                    print(
                        f"[OK] {ds_name} | k={k} | s={s} | li={li} | "
                        f"{selected_features} | time={train_time:.2f}s"
                    )

                    del feature_importances
                    gc.collect()

                # del query_dim_mat
                gc.collect()

        del all_queries_dask
        gc.collect()

    print(f"Finished training for {ds_name}")
    return results


def main():
    SEED = 42
    n_estimators = 100

    qid_feat = ["qid"]
    index_metric_feats = ["step", "dists", "inserts", "visited_points","duration"]
    neighbor_distances_feats = ["nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["mean_distance", "variance_of_distances"]

    all_feats = (
            qid_feat +
            index_metric_feats +
            neighbor_distances_feats +
            neighbor_stats_feats
    )

    columns_to_load = all_feats + ["r"]

    model_conf = {
        "raet": lgb.LGBMRegressor(
            objective="regression",
            random_state=SEED,
            n_estimators=n_estimators,
            verbose=-1
        )
    }

    feature_classes = {
        "all_feats": all_feats
    }

    dataset_params = {
        "SIFT1M": {"M": 32, "efC": 500, "efS": 500, "d": 128},
        "GIST1M": {"M": 32, "efC": 500, "efS": 1000, "d": 960},
        "GLOVE100": {"M": 16, "efC": 500, "efS": 500, "d": 100},
        "DEEP10M": {"M": 32, "efC": 500, "efS": 750, "d": 96},
    }

    training_queries_num = [10000]
    all_k_values = [100]
    all_li = [1]

    # for ds_name in ["SIFT1M", "GIST1M", "GLOVE100", "DEEP10M"]:
    for ds_name in ["SIFT1M"]:
    # for ds_name in ["DEEP10M"]:
        params = dataset_params[ds_name]

        results = train_predictors(
            ds_name,
            training_queries_num,
            all_k_values,
            all_li,
            params["M"],
            params["efC"],
            params["efS"],
            params["d"],
            n_estimators,
            columns_to_load,
            feature_classes,
            model_conf
        )

        out_path = f"/root/NSG-data/predictor_models/raet/k100_{ds_name}_qs10000.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        del results
        gc.collect()


if __name__ == "__main__":
    main()
