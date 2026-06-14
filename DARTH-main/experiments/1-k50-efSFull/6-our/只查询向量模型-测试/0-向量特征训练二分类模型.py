import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb


def read_fvecs(file_path, limit=None):
    """读取 fvecs 格式的向量文件"""
    with open(file_path, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        vector_size = (dim + 1) * 4
        file_size = os.fstat(f.fileno()).st_size
        num_vectors = file_size // vector_size

        if limit is not None:
            num_vectors = min(num_vectors, limit)

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)

        vectors = data[:, 1:]
        return vectors, dim


# ================== 全局配置 ==================
data_dir = "/home/extra_home/lxx23125236/ali/DARTH-main-data"

k = 50
s = 10000
SEED = 42
n_estimators = 100

# 不同数据集配置
DATASET_CONFIG = {
    "SIFT1M": {
        "d": 128,
        "efS": 500,
        "query_vec_path": "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_learn.fvecs"
    },
    "GIST1M": {
        "d": 960,
        "efS": 1000,
        "query_vec_path": "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_learn.fvecs"
    },
    "GLOVE100": {
        "d": 100,
        "efS": 500,
        "query_vec_path": "/home/extra_home/lxx23125236/ann-data/GLOVE100/learn.100K.fvecs"
    },
    "DEEP10M": {
        "d": 96,
        "efS": 750,
        "query_vec_path": "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_learn_10w_new.fvecs"
    }
}

# 需要训练的准确率
TR_LIST = [0.90, 0.92, 0.94, 0.96, 0.98]




def train_one_model(ds_name, tr):
    """训练单个数据集 + 单个准确率下的模型"""
    if ds_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    cfg = DATASET_CONFIG[ds_name]
    d = cfg["d"]
    efS = cfg["efS"]
    query_vec_path = cfg["query_vec_path"]

    # ================== 输入文件路径 ==================
    recall_label_file = (
        f"{data_dir}/result/raet/train/{ds_name}/k{k}/"
        f"efS{efS}_qs{s}_tr{tr:.2f}.txt"
    )
    cns_label_file = (
        f"{data_dir}/result/raet-cns-regression_CNSFull/train/{ds_name}/k{k}/"
        f"efS{efS}_qs{s}_tr{tr:.2f}.txt"
    )

    # ================== 输出路径 ==================
    model_output_dir = (
        f"{data_dir}/predictor_models/our/query_vec_only/{ds_name}/k{k}"
    )
    fi_output_dir = (
        f"{data_dir}/feature_importance/our/query_vec_only/{ds_name}/k{k}"
    )

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(fi_output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Start training: dataset={ds_name}, tr={tr:.2f}, d={d}, efS={efS}")

    # ================== 检查文件是否存在 ==================
    if not os.path.exists(query_vec_path):
        print(f"[Skip] Query vector file not found: {query_vec_path}")
        return

    if not os.path.exists(recall_label_file):
        print(f"[Skip] Recall label file not found: {recall_label_file}")
        return

    if not os.path.exists(cns_label_file):
        print(f"[Skip] CNS label file not found: {cns_label_file}")
        return

    # ================== 1. 读取查询向量 ==================
    print(f"Reading first {s} query vectors from {query_vec_path} ...")
    X_train, dim_read = read_fvecs(query_vec_path, limit=s)

    assert dim_read == d, f"Dimension mismatch: config={d}, file={dim_read}"
    assert X_train.shape == (s, d), f"Shape mismatch: expected ({s}, {d}), got {X_train.shape}"
    print(f"Query vectors loaded. Shape: {X_train.shape}")

    # ================== 2. 读取标签 ==================
    print("Loading labels ...")
    recall_label_df = pd.read_csv(recall_label_file)
    cns_label_df = pd.read_csv(cns_label_file)

    recall_time = recall_label_df["elaps_ms"].values
    cns_time = cns_label_df["elaps_ms"].values

    assert len(recall_time) == len(cns_time) == s, (
        f"Label length mismatch: recall={len(recall_time)}, cns={len(cns_time)}, expected={s}"
    )

    # 二分类标签: 0(CNS更快), 1(Recall更快)
    y_train = np.where(cns_time <= recall_time, 0, 1)

    num_total = len(y_train)
    num_0 = np.sum(y_train == 0)
    num_1 = np.sum(y_train == 1)

    print(f"Total samples: {num_total}")
    print(f"Label 0 (CNS faster): {num_0} ({num_0 / num_total:.4f})")
    print(f"Label 1 (Recall faster): {num_1} ({num_1 / num_total:.4f})")

    # ================== 3. 训练模型 ==================
    print("Training LightGBM model ...")
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=SEED,
        n_estimators=n_estimators,
        verbose=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # ================== 4. 保存特征重要性 ==================
    feature_names = [f"vec_dim_{i}" for i in range(d)]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    total_importance = importance_df["Importance"].sum()
    if total_importance > 0:
        importance_df["Importance_Ratio"] = importance_df["Importance"] / total_importance
    else:
        importance_df["Importance_Ratio"] = 0.0

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fi_file = (
        f"{fi_output_dir}/efS{efS}_qs{s}_{tr:.2f}_query_vec_importance.csv"
    )
    importance_df.to_csv(fi_file, index=False)
    print(f"Feature importance saved to: {fi_file}")

    # ================== 5. 保存模型 ==================
    model_file = (
        f"{model_output_dir}/efS{efS}_qs{s}_{tr:.2f}_query_vec_model.txt"
    )
    model.booster_.save_model(model_file)
    print(f"Model saved to: {model_file}")

    print(f"Training completed. Time taken: {train_time:.2f}s")


def main():
    for ds_name in DATASET_CONFIG.keys():
        for tr in TR_LIST:
            try:
                train_one_model(ds_name, tr)
            except Exception as e:
                print(f"[Error] dataset={ds_name}, tr={tr:.2f}, error={e}")


if __name__ == "__main__":
    main()