import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import time


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
        vectors = data[:, 1:]  # 去掉第一列的维度信息
        return vectors, dim


def total_seconds(avg_ms, q_num):
    return avg_ms * q_num / 1000.0


# ================== 全局配置 ==================
data_dir = "/home/extra_home/lxx23125236/ali/NSG-data"

k = 50
# query_num = 10000

DATASET_CONFIG = {
    "SIFT1M": {
        "d": 128,
        "efS": 500,
        "query_num":10000,
        "test_query_vec_path": "/home/extra_home/lxx23125236/ann-data/SIFT1M/sift_query.fvecs"
    },
    "GIST1M": {
        "d": 960,
        "efS": 1000,
        "query_num":1000,
        "test_query_vec_path": "/home/extra_home/lxx23125236/ann-data/GIST1M/gist_query.fvecs"
    },
    "GLOVE100": {
        "d": 100,
        "efS": 500,
        "query_num":10000,
        "test_query_vec_path": "/home/extra_home/lxx23125236/ann-data/GLOVE100/query.10K.fvecs"
    },
    "DEEP10M": {
        "d": 96,
        "efS": 750,
        "query_num":10000,
        "test_query_vec_path": "/home/extra_home/lxx23125236/ann-data/DEEP10M/deep10M_query_1wan.fvecs"
    }
}

TR_LIST = [0.90, 0.92, 0.94, 0.96, 0.98]


def evaluate_one_model(ds_name, tr):
    if ds_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    cfg = DATASET_CONFIG[ds_name]
    d = cfg["d"]
    efS = cfg["efS"]
    query_num = cfg["query_num"]
    query_vec_path = cfg["test_query_vec_path"]

    # ================== 路径配置 ==================
    model_input_dir = (
        f"{data_dir}/predictor_models/our/query_vec_only/{ds_name}/k{k}"
    )
    model_file = (
        f"{model_input_dir}/efS{efS}_s10000_{tr:.2f}_query_vec_model.txt"
    )

    recall_label_file = (
        f"{data_dir}/results/raet/{ds_name}/k{k}/"
        f"efS{efS}_s{query_num}_tr{tr:.2f}.txt"
    )
    cns_label_file = (
        f"{data_dir}/results/raet_CNS_regression/{ds_name}/k{k}/"
        f"efS{efS}_s{query_num}_tr{tr:.2f}.txt"
    )

    print("=" * 90)
    print(f"Evaluating dataset={ds_name}, tr={tr:.2f}, d={d}, efS={efS}")

    # ================== 文件存在性检查 ==================
    if not os.path.exists(model_file):
        print(f"[Skip] Model file not found: {model_file}")
        return None

    if not os.path.exists(query_vec_path):
        print(f"[Skip] Query vector file not found: {query_vec_path}")
        return None

    if not os.path.exists(recall_label_file):
        print(f"[Skip] Recall label file not found: {recall_label_file}")
        return None

    if not os.path.exists(cns_label_file):
        print(f"[Skip] CNS label file not found: {cns_label_file}")
        return None

    # ================== 1. 加载模型 ==================
    print(f"Loading model from: {model_file}")
    model = lgb.Booster(model_file=model_file)

    # ================== 2. 读取测试数据并预测 ==================
    print("Loading test queries and predicting...")
    X_test, test_dim = read_fvecs(query_vec_path, limit=query_num)

    assert test_dim == d, f"Dim mismatch: config={d}, file={test_dim}"
    assert X_test.shape == (query_num, d), (
        f"Shape mismatch: expected ({query_num}, {d}), got {X_test.shape}"
    )

    predict_start = time.time()
    y_pred_prob = model.predict(X_test)
    predict_time = time.time() - predict_start

    pred = (y_pred_prob > 0.5).astype(int)

    # ================== 3. 读取真实标签 ==================
    print("Loading ground-truth labels...")
    recall_df = pd.read_csv(recall_label_file)
    cns_df = pd.read_csv(cns_label_file)

    T_recall = recall_df["search_time"].values
    T_cns = cns_df["search_time"].values

    assert len(T_recall) == len(T_cns) == query_num, (
        f"Label length mismatch: recall={len(T_recall)}, cns={len(T_cns)}, expected={query_num}"
    )

    # 0: CNS更快, 1: Recall更快
    best = np.where(T_cns <= T_recall, 0, 1)

    # ================== 4. 准确率计算 ==================
    correct = (pred == best).sum()
    accuracy = correct / len(pred)

    # ================== 5. 时间分析 ==================
    selector_time_ideal = np.where(pred == 0, T_cns, T_recall)
    optimal_time = np.minimum(T_recall, T_cns)

    selector_total_s = total_seconds(selector_time_ideal.mean(), len(pred))
    optimal_total_s = total_seconds(optimal_time.mean(), len(pred))
    recall_total_s = total_seconds(T_recall.mean(), len(pred))
    cns_total_s = total_seconds(T_cns.mean(), len(pred))

    # ================== 6. 命令行输出 ==================
    print(f"Selector Accuracy: {accuracy:.4f} ({correct}/{len(pred)})")
    print("--- 时间比较如下： ---")
    print(f"选择器方法总时间(不含预测开销) (s): {selector_total_s:.2f}")
    print(f"最优情况选择器方法总时间   (s): {optimal_total_s:.2f}")
    print(f"Recall 方法总时间          (s): {recall_total_s:.2f}")
    print(f"CNS 方法总时间             (s): {cns_total_s:.2f}")

    return {
        "dataset": ds_name,
        "tr": tr,
        "accuracy": accuracy,
        "selector_total_s": selector_total_s,
        "optimal_total_s": optimal_total_s,
        "recall_total_s": recall_total_s,
        "cns_total_s": cns_total_s,
    }


def main():
    all_results = []

    for ds_name in DATASET_CONFIG.keys():
        for tr in TR_LIST:
            try:
                result = evaluate_one_model(ds_name, tr)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"[Error] dataset={ds_name}, tr={tr:.2f}, error={e}")

    # ================== 最后统一打印汇总 ==================
    print("\n" + "#" * 90)
    print("ALL RESULTS SUMMARY")
    print("#" * 90)

    if not all_results:
        print("No valid evaluation results generated.")
        return

    for item in all_results:
        print(
            f"[{item['dataset']}] tr={item['tr']:.2f} | "
            f"acc={item['accuracy']:.4f} | "
            f"f1_bin={item['f1_binary']:.4f} | "  # [新增]
            f"f1_mac={item['f1_macro']:.4f} | "   # [新增]
            f"avg_extra_time_ms={item['avg_extra_time_ms']:.4f} | "   # [新增]
            f"selector={item['selector_total_s']:.2f}s | "
            f"optimal={item['optimal_total_s']:.2f}s | "
            f"recall={item['recall_total_s']:.2f}s | "
            f"cns={item['cns_total_s']:.2f}s | "
        )


if __name__ == "__main__":
    main()