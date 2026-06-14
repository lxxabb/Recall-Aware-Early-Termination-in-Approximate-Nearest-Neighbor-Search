import os
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

# =========================================================
# 🔥 全局配置
# =========================================================
#k50,90,SIFT1M,GLOVE100,DEEP10M
datasets = ["GLOVE100", "GIST1M", "DEEP10M", "T2I1M"]
# datasets = ["SIFT1M"]
efs_map = {
    "SIFT1M": 500,
    "GLOVE100": 500,
    "GIST1M": 1000,
    "DEEP10M": 750,
    "T2I1M": 1000
}
k = 50
target_recall = "0.90"
SEED = 42
n_estimators = 100
all_features =  ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                   "mean_distance", "variance_of_distances","avg_hop","avg_indegree","entry_query_dist_ratio","dist_distribution_entropy",
                   "cosine_mean","cosine_variance","cosine_direction_entropy"]

# =========================================================
# 🔁 批量循环
# =========================================================
for dataset in datasets:
    print(f"\n================ {dataset} ================\n")
    efS = efs_map[dataset]
    # ===================== 路径 =====================
    # fea_file = f"/home/extra_home/lxx23125236/ali/NSG-data/data/et_training_data/raet-CNS-fea-training/{dataset}/k{k}/efS{efS}_qs20000.txt"
    #
    # recall_label_file = f"/home/extra_home/lxx23125236/ali/NSG-data/result/raet/train/{dataset}/k{k}/efS{efS}_qs20000_tr{target_recall}.txt"
    #
    # cns_label_file = f"/home/extra_home/lxx23125236/ali/NSG-data/result/raet-cns-regression_CNSFull/train/{dataset}/k{k}/efS{efS}_qs20000_tr{target_recall}.txt"

    fea_file = f"/home/extra_home/lxx23125236/ali/NSG-data/train_data/raet_CNS/{dataset}/k{k}/efS{efS}_qs20000.txt"

    recall_label_file = f"/home/extra_home/lxx23125236/ali/NSG-data/results/raet/train/{dataset}/k{k}/efS{efS}_s20000_tr{target_recall}.txt"

    cns_label_file = f"/home/extra_home/lxx23125236/ali/NSG-data/results/raet_CNS_regression/train/{dataset}/k{k}/efS{efS}_s20000_tr{target_recall}.txt"

    # ===================== 读取 =====================
    if not (os.path.exists(fea_file) and os.path.exists(recall_label_file) and os.path.exists(cns_label_file)):
        print(f"❌ 文件缺失，跳过 {dataset}")
        continue

    fea_df = pd.read_csv(fea_file)
    recall_label_df = pd.read_csv(recall_label_file)
    cns_label_df = pd.read_csv(cns_label_file)

    # ===================== 对齐 =====================
    merged_all = fea_df.merge(
        recall_label_df[['qid', 'search_time']],
        on='qid',
        how='inner'
    ).merge(
        cns_label_df[['qid', 'search_time']],
        on='qid',
        how='inner',
        suffixes=('_recall', '_cns')
    )

    merged_all = merged_all.sort_values('qid').reset_index(drop=True)
    X = merged_all[all_features].values

    recall_time = merged_all['search_time_recall'].values
    cns_time = merged_all['search_time_cns'].values

    assert len(X) == len(recall_time) == len(cns_time), "数据对齐失败"

    y = np.where(cns_time <= recall_time, 0, 1)

    print(f"✅ 样本数: {len(y)}")
    print(f"CNS更快: {np.sum(y==0)} | Recall更快: {np.sum(y==1)}")

    # =========================================================
    # ✅ 模型训练
    # =========================================================
    model = lgb.LGBMRegressor(
        objective='binary',
        random_state=SEED,
        n_estimators=n_estimators,
        verbose=-1
    )

    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start

    # ===================== 保存模型 =====================
    model_file = f"/home/extra_home/lxx23125236/ali/NSG-data/predictor_models/our/{dataset}/k{k}/efS{efS}_s20000_nestim100_selector_{target_recall}.txt"
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    model.booster_.save_model(model_file)

    print(f"✅ 模型保存: {model_file} | time={train_time:.2f}s")

    # ===================== 特征重要性 =====================
    fi = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    })

    fi_file = f"/home/extra_home/lxx23125236/ali/NSG-data/feature_importance/our/{dataset}/k{k}/fi_{target_recall}.csv"
    os.makedirs(os.path.dirname(fi_file), exist_ok=True)
    fi.to_csv(fi_file, index=False)

    # =========================================================
    # 🔥 决策树蒸馏（完全保留你的逻辑）
    # =========================================================
    print("\n===== Start Deep Decision Tree Distillation =====")

    X_small = merged_all[all_features].values

    X_train, X_val, y_train_split, y_val = train_test_split(
        X_small, y, test_size=0.2, random_state=42
    )

    def clean_export_tree(tree, feature_names):
        tree_ = tree.tree_
        lines = []

        def is_leaf(node):
            return tree_.children_left[node] == -1

        def get_final_class(node):
            if is_leaf(node):
                return np.argmax(tree_.value[node])
            return get_final_class(tree_.children_left[node])

        def recurse(node, depth):
            indent = "|   " * depth

            if is_leaf(node):
                pred = np.argmax(tree_.value[node])
                lines.append(f"{indent}|--- class: {pred}")
                return

            left = tree_.children_left[node]
            right = tree_.children_right[node]

            left_class = get_final_class(left)
            right_class = get_final_class(right)

            if left_class == right_class:
                lines.append(f"{indent}|--- class: {left_class}")
                return

            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            lines.append(f"{indent}|--- {feature} <= {threshold:.2f}")
            recurse(left, depth + 1)

            lines.append(f"{indent}|--- {feature} > {threshold:.2f}")
            recurse(right, depth + 1)

        recurse(0, 0)
        return "\n".join(lines)

    base_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,
        min_samples_split=100,
        random_state=42
    )
    base_tree.fit(X_train, y_train_split)

    path = base_tree.cost_complexity_pruning_path(X_train, y_train_split)
    ccp_alphas = path.ccp_alphas

    best_alpha = 0
    best_acc = 0
    best_tree = None

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=50,
            min_samples_split=100,
            ccp_alpha=alpha,
            random_state=42
        )

        clf.fit(X_train, y_train_split)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_tree = clf

    print(f"Best alpha: {best_alpha}")
    print(f"Validation Accuracy: {best_acc:.4f}")

    tree = best_tree

    # ===================== 剪枝 =====================
    def prune_invalid_splits(tree):
        tree_ = tree.tree_

        def is_leaf(node):
            return tree_.children_left[node] == _tree.TREE_LEAF

        def get_final_class(node):
            if is_leaf(node):
                return np.argmax(tree_.value[node])
            return get_final_class(tree_.children_left[node])

        def recurse(node):
            if is_leaf(node):
                return

            left = tree_.children_left[node]
            right = tree_.children_right[node]

            recurse(left)
            recurse(right)

            if get_final_class(left) == get_final_class(right):
                tree_.children_left[node] = _tree.TREE_LEAF
                tree_.children_right[node] = _tree.TREE_LEAF

        recurse(0)

    prune_invalid_splits(tree)

    # ===================== 规则 =====================
    rules = clean_export_tree(tree, all_features)

    if rules is None:
        print("⚠️ 规则为空，跳过")
        continue

    print("\n===== Clean Decision Tree Rules =====")
    print(rules[:2000])

    # ===================== 保存规则 =====================
    rule_file = f"/home/extra_home/lxx23125236/ali/NSG-data/result/our/{dataset}/k{k}/decision_tree_rules_{target_recall}.txt"
    os.makedirs(os.path.dirname(rule_file), exist_ok=True)

    with open(rule_file, "w") as f:
        f.write(rules)

    print(f"✅ 决策树规则保存: {rule_file}")