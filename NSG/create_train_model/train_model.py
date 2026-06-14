#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行训练 LightGBM 回归模型
usage:
    python train_lgb.py feat.csv learn.fvecs model.txt
"""
import os

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法
python train_lgb_mmap.py <K> <dim> <feat.csv> <learn.fvecs> <out_model.txt>
例：
python train_lgb_mmap.py 100 960 \
        train_fea.csv gist_learn_1wan.fvecs gist_lgb.txt
"""
# import sys
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# # ---------- 工具函数 ----------
# def ivecs_read(fname):
#     a = np.fromfile(fname, dtype='int32')
#     d = a[0]
#     return a.reshape(-1, d + 1)[:, 1:].copy()
#
# def fvecs_read(fname):
#     return ivecs_read(fname).view('float32')
#
# # ---------- 自定义 Dataset：按需拼接 ----------
# class LazyDataset(lgb.Dataset):
#     def __init__(self, base_feat, idx_vec, vec_mmap, label, **kw):
#         super().__init__(None, label=label, **kw)
#         self.base  = base_feat   # (n_row, 6)  统计特征
#         self.idv   = idx_vec     # (n_row,)    仅用于取 960 维向量
#         self.vec   = vec_mmap    # (10000, 960) memmap
#
#     def _construct_from_sample(self, row_idx):
#         base = self.base[row_idx]              # (batch, 6)
#         ids  = self.idv[row_idx]               # (batch,)
#         ext  = self.vec[ids]                   # (batch, 960)
#         return np.hstack([base, ext])          # (batch, 966)
#
# # ---------- 主函数 ----------
# def main(K: int, dim: int, feat_csv: str, learn_fvecs: str, model_out: str):
#     # 1. 读 CSV（index_id 已有序）
#     data = pd.read_csv(feat_csv, header=None)
#     columns = ["index_id", "CNS_current_index", "dist_0", "dist_k-1",
#                "mean_distance", "variance_of_distances",
#                "current_insert_count", "CNS_visited_points", "recall"]
#     data.columns = columns
#
#     # 2. 特征工程
#     data['CNS_visited_points'] /= K
#     label = data["recall"].to_numpy(np.float32)
#
#     # 3. 只保留 6 个统计特征（index_id 不进模型）
#     base_cols = ["dist_0", "dist_k-1", "mean_distance",
#                  "variance_of_distances", "current_insert_count",
#                  "CNS_visited_points"]
#     base_feat = data[base_cols].to_numpy(np.float32)   # (n_row, 6)
#     idx_vec   = data['index_id'].to_numpy(np.int32)    # 仅用于取向量
#
#     # # 4. memmap 映射 10 000×960 向量
#     # d = int(np.fromfile(learn_fvecs, dtype='int32', count=1)[0])
#     # vec_mmap = np.memmap(learn_fvecs, dtype='float32', mode='r',
#     #                      offset=4, shape=(10000, d))
#     # 0. 取维度
#     d = int(np.fromfile(learn_fvecs, dtype='int32', count=1)[0])
#
#     # 1. 用 memmap 跳过每行 4 字节维度字段
#     raw = np.memmap(learn_fvecs, dtype='int32', mode='r')
#     raw = raw.reshape(-1, d + 1)[:, 1:]      # 砍掉维度字段
#     vec_mmap = raw.view(np.float32)          # 无复制，形状 (10000, 960)
#
#     print(vec_mmap.shape)   # 应该输出 (10000, 960)
#     print(vec_mmap[0, 0])   # 应该输出一个正常浮点数，不是超大/超小值
#
#     # 5. 训练/验证划分（只用 6+960 列，index_id 不泄露）
#     train_idx, val_idx, y_train, y_val = train_test_split(
#         np.arange(len(data)), label, test_size=0.2, random_state=42)
#
#     # 6. 构造 Dataset——先 train 后 valid，且禁止提前释放
#     train_data = LazyDataset(base_feat, idx_vec, vec_mmap, y_train,
#                              free_raw_data=False)   # ← 新增
#     valid_data = LazyDataset(base_feat, idx_vec, vec_mmap, y_val,
#                              reference=train_data,   # ← 指定引用
#                              free_raw_data=False)    # ← 新增
#
#     # 7. 参数
#     params = {
#         'objective': 'regression',
#         'metric': ['l2', 'l1'],
#         'num_leaves': 31,
#         'learning_rate': 0.2,
#         'feature_fraction': 1.0,
#         'bagging_fraction': 1.0,
#         'verbose': 0,
#         'num_threads': 20,
#     }
#
#     # 8. 训练
#     model = lgb.train(params,
#                       train_data,
#                       num_boost_round=200,
#                       valid_sets=[train_data, valid_data],
#                       valid_names=['train', 'valid'],
#                       early_stopping_rounds=20,
#                       verbose_eval=10)
#
#     # 8. 评估
#     y_pred = model.predict(valid_data)
#     mae  = mean_absolute_error(y_val, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     r2   = r2_score(y_val, y_pred)
#     print(f"\nMAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
#
#     # 9. 特征重要性图（共 966 个，无 index_id）
#     feat_name = base_cols + [f'query_{i}' for i in range(dim)]
#     imp = pd.DataFrame({'feature': feat_name,
#                         'importance': model.feature_importance(importance_type='gain')})
#     imp = imp.sort_values('importance', ascending=False)
#     plt.figure(figsize=(8, 12))
#     imp.head(30).plot.barh(x='feature', y='importance', legend=False)
#     plt.tight_layout()
#     plt.savefig(model_out + '.png')
#     print(f"特征重要性图已保存：{model_out}.png")
#
#     # 10. 存模型
#     model.save_model(model_out)
#     print(f"模型已保存至：{model_out}")
#
# # ---------- 入口 ----------
# if __name__ == '__main__':
#     if len(sys.argv) != 6:
#         print("用法: python train_lgb_mmap.py <K> <dim> <feat.csv> <learn.fvecs> <out_model.txt>")
#         sys.exit(1)
#     K, dim = int(sys.argv[1]), int(sys.argv[2])
#     main(K, dim, sys.argv[3], sys.argv[4], sys.argv[5])
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')          # 无显示器环境也能出图
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 工具函数 ----------
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# ---------- 主函数 ----------
def main(K: int, dim: int, feat_csv: str, learn_fvecs: str, model_out: str):
    # 1. 读 CSV
    data = pd.read_csv(feat_csv, header=None)
    k = K
    columns = ["index_id", "CNS_current_index", "dist_0", "dist_k-1",
               "mean_distance", "variance_of_distances",
               "current_insert_count", "CNS_visited_points", "recall"]
    data.columns = columns

    # 2. 特征工程
    data['CNS_visited_points'] = data['CNS_visited_points'] / k

    label = data["recall"]
    selected = ["index_id", "dist_0", "dist_k-1", "mean_distance",
                "variance_of_distances", "current_insert_count",
                "CNS_visited_points"]
    feature_important_name=["dist_0", "dist_k-1", "mean_distance","variance_of_distances", "current_insert_count","CNS_visited_points"]
    selected_data = data[selected]
    del data

    # # 3. 拼接查询向量
    learn_mat = fvecs_read(learn_fvecs)          # shape = (10000, 128)
    q_df = pd.DataFrame(learn_mat, columns=[f'query_{i}' for i in range(dim)])
    q_df.index.name = 'index_id'

    # 拼好真实列名, 方便输出特征重要性时得到真实的列名
    learn_dim = learn_mat.shape[1]                  # 128
    learn_cols = [f'v{i}' for i in range(learn_dim)]  # 查询向量列名，可自定义
    all_cols = feature_important_name + learn_cols                 # 7 + 128 = 135

    del learn_mat
    all_data = (selected_data.set_index('index_id')
            .merge(q_df, left_index=True, right_index=True, how='left')
            .reset_index()
            .drop(columns='index_id'))
    del q_df
    import gc
    gc.collect()

    # ===== 4. 统一采样 1 000 000 行 =====
    SAMPLE_NUM = 1_000_000
    if len(all_data) > SAMPLE_NUM:
        # 若想固定随机种子，多加 random_state=42
        np.random.seed(42)          # 固定全局随机种子
        idx = np.random.choice(len(all_data), size=SAMPLE_NUM, replace=False)
        all_data = all_data.iloc[idx].reset_index(drop=True)
        label    = label.iloc[idx].reset_index(drop=True)

    # 4. 训练/验证划分
    # X_train, X_val, y_train, y_val = train_test_split(all_data, label, test_size=0.2, shuffle=True, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(
    #     all_data.values.astype(np.float32),
    #     label.values.astype(np.float32),
    #     test_size=0.2, random_state=42)
    # del all_data, label
    # gc.collect()
    #
    # train_data = lgb.Dataset(X_train, y_train)
    # valid_data = lgb.Dataset(X_val, y_val, reference=train_data)
    X_train, X_val, y_train, y_val = train_test_split(
        all_data, label, test_size=0.2, shuffle=True, random_state=42)

    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_val, y_val, reference=train_data)

    # 5. 参数与训练
    #gist数据集使用如下
    # if dim==960:
    #     params = {
    #         'objective': 'regression',
    #         'metric': {'l2'},
    #         'boosting_type': 'gbdt',
    #         'learning_rate': 0.20,
    #         'num_leaves': 20,          # 关键：20 叶子
    #         'max_depth': 6,
    #         'feature_fraction': 0.8,   # 轻微列采样，防过拟合
    #         'bagging_fraction': 0.8,
    #         'bagging_freq': 1,
    #         'min_data_in_leaf': 6,
    #         'max_bin': 63,             # 直方图量化
    #         'lambda_l2': 0.1,
    #         'verbose': -1,
    #     }
    #     evals_result = {}
    #     model = lgb.train(params,
    #                       train_data,
    #                       num_boost_round=300,
    #                       valid_sets=[train_data, valid_data],
    #                       valid_names=['train', 'valid'],
    #                       early_stopping_rounds=20,
    #                       evals_result=evals_result,
    #                       verbose_eval=10)
    #
    #
    # #sift数据集使用如下即可
    # else:
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'boost_from_average': False,
        'learning_rate': 0.2,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'verbose': 0,
        'num_threads': 1,
    }
    evals_result = {}
    model = lgb.train(params,
                  train_data,
                  num_boost_round=200,
                  valid_sets=[train_data, valid_data],
                  valid_names=['train', 'valid'],
                  early_stopping_rounds=20,
                  evals_result=evals_result,
                  verbose_eval=10)

    # 6. 评估
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"\nMAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

    #特征重要性
    def plot_imp(imp_type, png_path):
        imp = model.feature_importance(importance_type=imp_type)
        feat_imp = pd.DataFrame({'feature': all_cols, 'importance': imp})
        # 计算占比
        feat_imp['pct'] = feat_imp['importance'] / feat_imp['importance'].sum() * 100

        # 2. 全部特征写盘（按重要性降序）
        csv_path = f"{model_out}_{imp_type}.csv"
        feat_imp.sort_values('pct', ascending=False).to_csv(csv_path,index=False,encoding='utf-8-sig')
        print(f'全部特征重要性及占比，已写入csv文件：{csv_path}')

        feat_imp = feat_imp.sort_values('pct', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        # 水平条形图，x 轴直接是百分比
        plt.barh(feat_imp['feature'], feat_imp['pct'], color='steelblue')
        plt.gca().invert_yaxis()          # 让最高的在上面
        plt.xlabel('Importance (%)')
        plt.title(f'Top 20 Feature Importance ({imp_type}) — Percentage')
        plt.rcParams['figure.dpi'] = 300          # 分辨率
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 14            # 全局字号
        # 在柱子上显示百分比
        for i, v in enumerate(feat_imp['pct']):
            plt.text(v + 0.1, i, f'{v:.1f}%', va='center')
        plt.tight_layout()
        plt.savefig(png_path)
        print(f"前20占比特征重要性占比图已保存：{png_path}")

    plot_imp('split', f"{model_out}_split.png")
    plot_imp('gain',  f"{model_out}_gain.png")

    # 8. 保存模型
    model.save_model(model_out)
    print(f"模型已保存至：{model_out}")

# ---------- 入口 ----------
if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("用法: python train_lgb.py <K> <dim> <features.csv> <learn.fvecs> <output_model.txt>")
        sys.exit(1)
    K   = int(sys.argv[1])
    dim = int(sys.argv[2])
    main(K, dim, sys.argv[3],sys.argv[4],sys.argv[5])