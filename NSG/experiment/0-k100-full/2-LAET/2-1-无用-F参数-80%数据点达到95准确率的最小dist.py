import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import time
import dask.dataframe as dd

SEED = 42
#该脚本从 early-stop 搜索日志中构建 query-level 训练样本，利用搜索过程中早期统计特征与 query 向量本身作为输入，
# 训练 LightGBM 回归模型预测达到最大 recall 所需的最小距离计算量，从而支持精细化的 query-adaptive early termination。
def get_dataset_name(M, efC, efS, query_num, ds_name, k):
    return f"/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/{ds_name}/k{k}/efS{efS}_qs{query_num}.txt"

# def get_optimal_dataset_name(M, ef, s, ds_name, k, logint):
#     return f"../results/early-stop-training/RECALL_{ds_name}_M{M}_ef{ef}_k{k}_qs{s}_li{logint}.txt"

def read_bvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected bvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = 4 + dim

        file_size = f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        num_vectors = file_size // vector_size

        #print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")

        if (limit is not None) and limit < num_vectors:
            num_vectors = limit
            #print(f">>Limiting to {num_vectors} vectors")

        data = np.fromfile(f, dtype=np.uint8, count=num_vectors * vector_size)
        data = data.reshape(-1, vector_size)

        vectors = data[:, 4:]

        #print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)

    return vectors, int(dim)

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

def read_fbin_vecs(file_path, limit=None):
    vectors = None
    dim = None

    with open(file_path, 'rb') as f:
        n = np.fromfile(f, count=1, dtype=np.uint32)[0]
        dim = np.fromfile(f, count=1, dtype=np.uint32)[0]

        if limit is not None and n > limit:
            n = limit
            #print(f">> Limiting dataset {file_path} to {n} vectors")

        vectors = np.fromfile(f, count=n * dim, dtype=np.float32)
        vectors = vectors.reshape(n, dim)

    return vectors, int(dim)

def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def get_query_dims_df(data_df, d, ds_name, s):
    dimensions = np.arange(1, d + 1)
    per_query_dimensions_df = pd.DataFrame(index=data_df["qid"].unique(), columns=[f"d{d}" for d in dimensions])

    query_dims = None
    dims_read = None

    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ali/ann-data/SIFT1M/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ali/ann-data/GIST1M/gist_learn.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ali/ann-data/GLOVE100/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP10M":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ali/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs", limit=s)
    elif ds_name == "T2I1M":
        query_dims, dims_read = read_fvecs(f"/home/extra_home/lxx23125236/ann-data/T2I1M/query.heldout.30K.train2w.fvecs", limit=s)

    assert query_dims.shape[0] == s, f"Expected {s} queries, got {query_dims.shape[0]}"
    assert query_dims.shape[1] == d, f"Expected {d} dimensions, got {query_dims.shape[1]}"
    assert dims_read == d, f"Expected {d} dimensions, got {dims_read}"

    # Make sure that this is correct
    for i, qid in enumerate(data_df["qid"].unique()):
        per_query_dimensions_df.loc[qid] = query_dims[i]

        #assert len(per_query_dimensions_df) == s, f"Expected {s} queries, got {len(per_query_dimensions_df)}"

    return per_query_dimensions_df, dimensions

def main():
    s, li = 10000, 1

    dataset_info = {
        "SIFT1M":{
            "M": 32,
            "efC": 500,
            "efS": 500,
            "d": 128,
            "F": 241
        },
        "GIST1M": {
            "M": 32,
            "efC": 500,
            "efS": 1000,
            "d": 960,
            "F": 1260,
        },
        "GLOVE100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "F": 200,
        },
        "DEEP10M": {
            "M": 32,
            "efC": 500,
            "efS": 750,
            "d": 96,
            "F": 368,
        },
        "T2I1M": {
            "M": 80,
            "efC": 1000,
            "efS": 1000,
            "d": 200,
            "F": 1000,
        },
    }

    for k in [100]:
        # for ds_name in ["SIFT1M","GIST1M","GLOVE100","DEEP10M"]:
        for ds_name in ["T2I1M"]:
            print (f"Training for {ds_name} with k={k}")

            d = dataset_info[ds_name]["d"]
            F = dataset_info[ds_name]["F"]
            M = dataset_info[ds_name]["M"]
            efC = dataset_info[ds_name]["efC"]
            efS = dataset_info[ds_name]["efS"]

            # Load all the data
            columns_to_load = ["qid", "dists", "first_nn_dist", "nn_dist", "nn10_dist", "nn_to_first", "nn10_to_first", "r"]
            dask_df = pd.read_csv(get_dataset_name(M, efC, efS, 10000, ds_name, k), usecols=columns_to_load,dtype={'r': 'float64','nn_to_first': 'float64'})
            TARGET_RECALL = 0.90
            df_ok = dask_df[dask_df["r"] >= TARGET_RECALL]
            min_dists_per_qid = (
                df_ok
                .groupby("qid")["dists"]
                .min()
            )
            dist_80 = min_dists_per_qid.quantile(0.5)
            dist_80 = dist_80.item()
            print(f"80% queries 达到 0.95 准确率所需的最小距离计算次数: {dist_80:.2f}")


if __name__ == "__main__":
    main()
# Training for SIFT1M with k=100
# 80% queries 达到 0.95 准确率所需的最小距离计算次数: 3553.00
# Training for GIST1M with k=100
# 80% queries 达到 0.95 准确率所需的最小距离计算次数: 11452.40
# Training for GLOVE100 with k=100
# 80% queries 达到 0.95 准确率所需的最小距离计算次数: 1342.00
# Training for DEEP10M with k=100
# 80% queries 达到 0.95 准确率所需的最小距离计算次数: 5234.60