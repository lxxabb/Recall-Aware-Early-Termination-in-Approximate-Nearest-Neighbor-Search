import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
import dask.dataframe as dd


def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"/root/DARTH-main-data/data/et_training_data/early-stop-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}_new.txt"


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


def train_predictors(ds_name, training_queries_num, all_k_values, all_li,
                     M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf):

    results = {}
    max_query_size = max(training_queries_num)
    print(f"Starting training for {ds_name}")

    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, li)
        print(f"{ds_name} | k={k} | Path: {data_path}")

        all_queries_dask = dd.read_csv(
            data_path,
            usecols=columns_to_load,
            blocksize="128MB"
        )

        for s in training_queries_num:
            results[k][s] = {}

            query_data_all = all_queries_dask[all_queries_dask["qid"] < s]

            for li in all_li:
                results[k][s][li] = {}

                data_all = query_data_all[query_data_all["dists"] % li == 0]

                # 只 compute label
                y_all = data_all["r"].compute()

                for selected_features in feature_classes:
                    feats = feature_classes[selected_features]

                    # 只 compute 需要的列
                    data_df = data_all[feats].compute()
                    #将visited_points变为与k的比值特征
                    data_df["visited_points"] = data_df["visited_points"] / k
                    input_features=[
                        "step", "dists", "inserts",
                        "visited_points",
                        "duration",
                        "nn_dist",
                        "furthest_dist",
                        "avg_dist",
                        "variance"
                    ]
                    X_train = data_df[[
                        "step", "dists", "inserts",
                        "visited_points",
                        "duration",
                        "nn_dist",
                        "furthest_dist",
                        "avg_dist",
                        "variance"
                    ]].values  # float64 保留

                    # X_train = np.hstack([X_core, query_embed])

                    model = model_conf["raet"]
                    t0 = time.time()
                    model.fit(X_train, y_all)
                    train_time = time.time() - t0

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
                    fi_file = f"/root/DARTH-main-data/feature_importance/raet/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}_Noquery_duration.csv"
                    # 保存到文件
                    os.makedirs(os.path.dirname(fi_file), exist_ok=True)
                    feature_importances.to_csv(fi_file, index=False)
                    print(f"特征重要性保存到: {fi_file}")

                    model_file = f"/root/DARTH-main-data/predictor_models/raet/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}_Noquery_duration.txt"
                    os.makedirs(os.path.dirname(model_file), exist_ok=True)
                    model.booster_.save_model(model_file)

                    results[k][s][li][selected_features] = {
                        "training_time": train_time,
                        "training_data_size": X_train.shape[0],
                        "n_estimators": n_estimators
                    }

    print(f"Finished training for {ds_name}")
    return results


def main():
    SEED = 42
    n_estimators = 100

    qid_feat = ["qid"]
    index_metric_feats = ["step", "dists", "inserts", "visited_points","duration"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance"]
    all_feats = qid_feat + index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

    columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

    model_conf = {
        "raet": lgb.LGBMRegressor(objective="regression", random_state=SEED,
                                  n_estimators=n_estimators, verbose=-1)
    }

    feature_classes = {
        "all_feats": all_feats
    }

    dataset_params = {
        "SIFT1M": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d":128,
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
            "d": 100,
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
        },
    }

    training_queries_num = [10000]
    all_k_values = [100]
    all_datasets = ["SIFT1M","GIST1M", "GLOVE100", "DEEP10M"]
    # all_datasets = ["SIFT1M"]
    all_li = [1]

    for ds_name in all_datasets:
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

        out_path = f"/root/DARTH-main-data/predictor_models/raet/{ds_name}_qs10000.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
