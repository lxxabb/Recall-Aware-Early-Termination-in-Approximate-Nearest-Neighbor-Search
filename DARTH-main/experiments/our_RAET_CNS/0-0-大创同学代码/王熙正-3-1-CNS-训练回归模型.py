import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.dataframe as dd

# ==========================================
# 请把这段加在 import 之后，代码的最上面
# ==========================================
# 你的数据似乎在这里：D:\Learnnnn\dachuang\DARTH-main-data
BASE_DIR = r"D:\Learnnnn\dachuang\DARTH-main-data" 
ANN_DIR = r"D:\Learnnnn\dachuang\ann-data"  # 假设你的向量文件在这里
# ==========================================


# ==========================================
#  新增：方案一 AdaptiveBoundLGBM 类定义
# ==========================================
class AdaptiveBoundLGBM(BaseEstimator, RegressorMixin):
    def __init__(self, min_val, max_val, n_estimators=100, learning_rate=0.1, 
                 k_init=1.0, k_learning_rate=0.01, random_state=42):
        """
        Scheme 1: Loss Function with Trainable Parameter
        """
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.k = k_init
        self.k_lr = k_learning_rate
        self.random_state = random_state
        self.booster_ = None
        self.feature_importances_ = None # 用于兼容后续的特征重要性保存

    def _sigmoid(self, x):
        # 截断防止溢出
        x = np.clip(x, -50, 50) 
        return 1 / (1 + np.exp(-x))

    def _get_range(self):
        return self.max_val - self.min_val

    def custom_objective(self, y_true, y_pred_raw):
        # LightGBM 传进来的 y_pred_raw 是 z (原始分值)
        R = self._get_range()
        z = y_pred_raw
        
        # 1. 前向传播
        S = self._sigmoid(self.k * z)
        y_hat = self.min_val + R * S
        
        # 2. 残差
        residual = y_hat - y_true
        
        # 3. 梯度 (针对 z)
        # dy_hat / dz = R * k * S * (1-S)
        grad_factor = R * self.k * S * (1 - S)
        grad = residual * grad_factor
        
        # 4. Hessian (Gauss-Newton)
        hess = grad_factor ** 2
        
        return grad, hess

    def fit(self, X, y):
        train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        
        # 2. 定义目标函数 wrapper
        # LightGBM 要求的格式: (preds, dataset) -> (grad, hess)
        def custom_obj_func(preds, train_data):
            return self.custom_objective(train_data.get_label(), preds)

        # 3. 准备参数字典
        # 【核心修改】：直接将函数对象赋值给 params['objective']
        # 这样就不需要在 lgb.train 里传 fobj 参数了，避开报错
        params = {
            'learning_rate': self.learning_rate,
            'verbose': -1,
            'metric': 'None',
            'seed': self.random_state,
            'num_threads': 4,
            'objective': custom_obj_func  # <--- 关键在这里！
        }
        
        self.booster_ = None
        
        # 4. 手动 Boosting 循环
        print(f"开始训练 AdaptiveBoundLGBM... 初始 k={self.k:.4f}")
        
        for i in range(self.n_estimators):
            # Step 1: 训练一轮树 (固定 k, 更新 z)
            self.booster_ = lgb.train(
                params,
                train_data,
                num_boost_round=1,
                init_model=self.booster_,
                # 注意：这里删除了 fobj=... 和 objective=... 
                # 因为已经在 params 里定义了
                keep_training_booster=True
            )
            
            # Step 2: 更新 k (固定 z, 更新 k)
            z_current = self.booster_.predict(X, raw_score=True)
            
            R = self._get_range()
            S = self._sigmoid(self.k * z_current)
            y_hat = self.min_val + R * S
            
            # dL/dk
            grad_k_all = (y_hat - y) * (R * z_current * S * (1 - S))
            grad_k_mean = np.mean(grad_k_all)
            
            # 更新 k
            self.k -= self.k_lr * grad_k_mean
            self.k = max(0.001, self.k)
        
        # 训练结束后获取特征重要性
        self.feature_importances_ = self.booster_.feature_importance(importance_type='gain')
        return self

# ==========================================
#  原有的辅助函数
# ==========================================

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, logint):
    return f"{BASE_DIR}/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/efS{efS}_qs{num_queries}.txt"
def get_label_dataset_name(ds_name, k,target_recall,num_queries):
    return f"{BASE_DIR}/data/et_training_data/raet-CNS-fea-training/{ds_name}/k{k}/min_ef_per_query_s{num_queries}_{target_recall}.csv"

def read_fvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected fvecs format.")
        dim = first_entry[0]

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
        vectors = data[:, 1:]
        assert vectors.shape == (num_vectors, dim)

    return vectors, int(dim)

def get_query_dims_df(data_df, d, ds_name, s):
    dimensions = np.arange(1, d + 1)
    per_query_dimensions_df = pd.DataFrame(index=data_df["qid"].unique(), columns=[f"d{d}" for d in dimensions])

    if ds_name == "SIFT1M":
        query_dims, dims_read = read_fvecs(f"{ANN_DIR}/SIFT1M/sift_learn.fvecs", limit=s)
    elif ds_name == "GIST1M":
        query_dims, dims_read = read_fvecs(f"{ANN_DIR}/GIST1M/gist_learn.fvecs", limit=s)
    elif ds_name == "GLOVE100":
        query_dims, dims_read = read_fvecs(f"{ANN_DIR}/GLOVE100/learn.100K.fvecs", limit=s)
    elif ds_name == "DEEP10M":
        query_dims, dims_read = read_fvecs(f"{ANN_DIR}/DEEP10M/deep10M_learn_10w_new.fvecs", limit=s)

    assert query_dims.shape[0] == s
    assert dims_read == d

    for i, qid in enumerate(data_df["qid"].unique()):
        per_query_dimensions_df.loc[qid] = query_dims[i]

    return per_query_dimensions_df, dimensions

def get_CNS(ds_name,target_recall):
    if ds_name == "SIFT1M":
        if target_recall==0.99: return 172
        if target_recall==0.98: return 128
        if target_recall==0.97: return 107
        if target_recall==0.96: return 100
        if target_recall==0.95: return 100
    if ds_name == "GLOVE100":
        if target_recall==0.99: return 247
        if target_recall==0.98: return 116
        if target_recall==0.97: return 100
        if target_recall==0.96: return 100
        if target_recall==0.95: return 100
    if ds_name == "GIST1M":
        if target_recall==0.99: return 788
        if target_recall==0.98: return 503
        if target_recall==0.97: return 385
        if target_recall==0.96: return 316
        if target_recall==0.95: return 270
    if ds_name == "DEEP10M":
        if target_recall==0.99: return 271
        if target_recall==0.98: return 181
        if target_recall==0.97: return 141
        if target_recall==0.96: return 117
        if target_recall==0.95: return 100
    return 100 # Default

# ==========================================
#  修改后的 train_predictors 函数
# ==========================================
def train_predictors(ds_name, training_queries_num, all_k_values, all_li, M, efC, efS, d, n_estimators, columns_to_load, feature_classes, model_conf,target_recalls):
    results = {}
    SEED = 42

    max_query_size = max(training_queries_num)

    print(f"Starting training for {ds_name} [SCHEME 1: AdaptiveBound]")

    for k in all_k_values:
        results[k] = {}
        li = 1

        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, li)
        print(f"{ds_name} | k={k} | Path: {data_path}")

        all_queries_pd = pd.read_csv(data_path, usecols=columns_to_load)

        for s in training_queries_num:
            for target_recall in target_recalls:
                CNS = get_CNS(ds_name,target_recall)
                if CNS==k:
                    continue
                
                print(f"Target Recall: {target_recall} | Bounds: [{k}, {CNS}]")
                results[k][s] = {}
                query_data_all = all_queries_pd[all_queries_pd["qid"] < s]
                qid_pd = query_data_all[["qid"]]
                per_query_dimensions_df, dimensions = get_query_dims_df(qid_pd, d, ds_name, s)

                label_path = get_label_dataset_name(ds_name, k,target_recall,max_query_size)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Min CNS file not found: {label_path}")
                min_cns_df = pd.read_csv(label_path)
                min_cns_df.set_index("qid", inplace=True)

                for li in all_li:
                    results[k][s][li] = {}
                    data_all = query_data_all[query_data_all["dists"] % li == 0]

                    print(f"    {s} Queries | Li: {li} | Shape: {data_all.shape}")

                    y = data_all["qid"].map(lambda qid: min_cns_df.loc[qid, "min_efSearch"])

                    # ---------------------------------------------------------
                    # [Scheme 1 关键步骤] 
                    # 标签处理：必须截断在 [min_val, max_val] 之间，否则Sigmoid无法拟合
                    # ---------------------------------------------------------
                    y_all = np.clip(y, k, CNS)

                    for selected_features in feature_classes:
                        feats = feature_classes[selected_features]

                        data_df = data_all[feats]
                        per_query_dimensions_pd = per_query_dimensions_df.copy()
                        data_df = data_df.merge(per_query_dimensions_pd,left_on="qid",right_index=True)

                        query_dims_features = [f"d{d}" for d in dimensions]
                        input_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                                           "avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy",
                                           "cosine_mean","cosine_variance","cosine_direction_entropy"] + query_dims_features

                        X_train = data_df[input_features].values
                        y_train = y_all

                        # ---------------------------------------------------------
                        # [Scheme 1 关键步骤] 实例化自定义模型
                        # ---------------------------------------------------------
                        model = AdaptiveBoundLGBM(
                            min_val=k,      # a
                            max_val=CNS,    # b
                            n_estimators=n_estimators,
                            learning_rate=0.05, # 树的学习率
                            k_init=0.5,         # k 初始值
                            k_learning_rate=0.01, # k 学习率
                            random_state=SEED
                        )

                        model_train_time_start = time.time()
                        model.fit(X_train, y_train) # 这里会运行 custom loop
                        model_train_time = time.time() - model_train_time_start

                        # 获取特征重要性 (AdaptiveBoundLGBM 类中已封装)
                        feature_importances = pd.DataFrame({
                            'Feature': input_features,
                            'Importance': model.feature_importances_
                        })
                        total_importance = feature_importances['Importance'].sum()
                        feature_importances['Importance_Ratio'] = feature_importances['Importance'] / total_importance
                        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

                        # 保存特征重要性
                        fi_file = f"{BASE_DIR}/feature_importance/raet-cns-regression/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}_{target_recall}_Scheme1.csv"
                        os.makedirs(os.path.dirname(fi_file), exist_ok=True)
                        feature_importances.to_csv(fi_file, index=False)

                        # ---------------------------------------------------------
                        # [Scheme 1 关键步骤] 保存模型 AND 参数
                        # ---------------------------------------------------------
                        base_path = f"{BASE_DIR}/predictor_models/raet-cns-regression/{ds_name}/k{k}/efS{efS}_s{s}_nestim{n_estimators}_{selected_features}_{target_recall}_Scheme1"
                        model_file = base_path + ".txt"
                        param_file = base_path + "_params.json"
                        
                        os.makedirs(os.path.dirname(model_file), exist_ok=True)
                        model.booster_.save_model(model_file)
                        
                        # 必须手动保存学到的参数，验证时要用
                        params_to_save = {
                            "final_k": float(model.k),
                            "min_val": float(model.min_val),
                            "max_val": float(model.max_val)
                        }
                        with open(param_file, "w") as f:
                            json.dump(params_to_save, f, indent=4)

                        print(f"        Train Time: {model_train_time:.2f} | Final k: {model.k:.4f} | Saved: {model_file}")

                        results[k][s][li][selected_features] = {
                            "training_time": model_train_time,
                            "training_data_size": X_train.shape[0],
                            "final_k": model.k,
                            "n_estimators": n_estimators,
                        }

                print()
        print("\n\n")
    print(f"Finished training for {ds_name}")

    return results

def main():
    import lightgbm
    print(f"DEBUG: LightGBM Version: {lightgbm.__version__}")
    print(f"DEBUG: LightGBM File: {lightgbm.__file__}")

    SEED = 42
    n_estimators = 100
    
    qid_feat=["qid"]
    index_metric_feats = ["step", "dists", "inserts"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance","avg_hop","density","entry_query_dist_ratio","dist_distribution_entropy","skewness","kurtosis","energy"]
    cosine_feats = ["cosine_mean","cosine_variance","cosine_direction_entropy"]
    all_feats = qid_feat+ index_metric_feats + neighbor_distances_feats + neighbor_stats_feats + cosine_feats
    columns_to_load = ["elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]

    # 这里的 model_conf 已经没用了，但为了保持参数一致性保留占位符
    model_conf = {} 

    feature_classes = {
        "all_feats": all_feats,
    }

    dataset_params = {
        "SIFT1M": { "M": 32, "efC": 500, "efS": 500, "d":128 },
        "GIST1M": { "M": 32, "efC": 500, "efS": 1000, "d": 960 },
        "GLOVE100": { "M": 16, "efC": 500, "efS": 500, "d": 100 },
        "DEEP10M": { "M": 32, "efC": 500, "efS": 750, "d": 96 },
    }

    training_queries_num = [10000] # 可以根据需要改回 [100000]
    all_k_values = [100]
    all_datasets = ["GLOVE100"]
    all_li = [1]
    target_recalls = [0.95,0.96,0.97,0.98,0.99]

    for ds_name in all_datasets:
        M = dataset_params[ds_name]["M"]
        efC = dataset_params[ds_name]["efC"]
        efS = dataset_params[ds_name]["efS"]
        d = dataset_params[ds_name]["d"]
        results = train_predictors(
            ds_name, training_queries_num, all_k_values, all_li,
            M, efC, efS, d,n_estimators, columns_to_load, feature_classes, model_conf,target_recalls)

        with open(f"{BASE_DIR}/predictor_models/raet-cns-regression/{ds_name}_qs10000_Scheme1_log.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()