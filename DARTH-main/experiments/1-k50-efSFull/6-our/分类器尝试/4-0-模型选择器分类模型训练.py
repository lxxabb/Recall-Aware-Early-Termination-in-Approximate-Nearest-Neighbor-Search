#特征路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k100/efS500_qs10000.txt
#recall标签路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/train/SIFT1M/k100/efS500_qs10000_tr0.95_60.txt
#CNS标签路径：/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/train/SIFT1M/k100/efS500_qs10000_tr0.95.txt
import os
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time
import json
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
fea_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/SIFT1M/k50/efS500_qs20000.txt"
recall_label_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/train/SIFT1M/k50/efS500_qs20000_tr0.90.txt"
cns_label_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet-cns-regression_CNSFull/train/SIFT1M/k50/efS500_qs20000_tr0.90.txt"

# ===================== 读取文件 =====================
fea_df = pd.read_csv(fea_file)
recall_label_df = pd.read_csv(recall_label_file)
cns_label_df = pd.read_csv(cns_label_file)

# ===================== 核心：按 qid 取交集 =====================
# 1. 先把 recall 和 cns 标签按 qid 内连接，只保留共同 qid
merged_labels = pd.merge(
    recall_label_df[['qid', 'elaps_ms']],
    cns_label_df[['qid', 'elaps_ms']],
    on='qid',  # 按 qid 匹配
    suffixes=('_recall', '_cns'),
    how='inner'  # 只保留两个文件都有的 qid
)

# 2. 特征文件也只保留共同 qid
merged_final = pd.merge(
    fea_df,
    merged_labels[['qid']],
    on='qid',
    how='inner'
)

# 3. 按 qid 排序，保证顺序一致
merged_final = merged_final.sort_values('qid').reset_index(drop=True)
merged_labels = merged_labels.sort_values('qid').reset_index(drop=True)

# 4. 提取对齐后的数据
X_train = merged_final[["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                        "avg_dist", "variance","avg_hop","entry_query_dist_ratio","dist_distribution_entropy",
                        "cosine_mean","cosine_variance","cosine_direction_entropy"]].values

recall_time = merged_labels['elaps_ms_recall'].values
cns_time = merged_labels['elaps_ms_cns'].values

# ===================== 检查对齐完成 =====================
assert len(X_train) == len(recall_time) == len(cns_time), "对齐后数据长度不一致！"
print(f"✅ 共找到 {len(X_train)} 个相同 qid 的训练样本")

# ===================== 生成二分类标签 =====================
# cns更快 -> 0
# recall更快 -> 1
y_all = np.where(cns_time <= recall_time, 0, 1)

num_total = len(y_all)
num_0 = np.sum(y_all == 0)
num_1 = np.sum(y_all == 1)

print(f"Total samples: {num_total}")
print(f"Label 0 (CNS faster): {num_0} ({num_0 / num_total:.4f})")
print(f"Label 1 (Recall faster): {num_1} ({num_1 / num_total:.4f})")

y_train = y_all

# ===================== 模型训练 =====================
SEED = 42
n_estimators = 100
model_conf = {
    "raet-cns": lgb.LGBMRegressor(objective='binary', random_state=SEED, n_estimators=n_estimators, verbose = -1),
}
model = model_conf["raet-cns"]
model_train_time_start = time.time()
model.fit(X_train, y_train)
model_train_time = time.time() - model_train_time_start

model_params = model.get_params()
learning_rate = model_params["learning_rate"]

# ===================== 特征重要性 =====================
feature_importances = pd.DataFrame({
    'Feature': ["step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
                "avg_dist", "variance","avg_hop","entry_query_dist_ratio","dist_distribution_entropy",
                "cosine_mean","cosine_variance","cosine_direction_entropy"],
    'Importance': model.feature_importances_
})

total_importance = feature_importances['Importance'].sum()
feature_importances['Importance_Ratio'] = feature_importances['Importance'] / total_importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

fi_file = f"/home/extra_home/lxx23125236/ali/DARTH-main-data/feature_importance/our/SIFT1M/k50/efS500_s10000_nestim100_all_feats_recall_selector_0.90.csv"
os.makedirs(os.path.dirname(fi_file), exist_ok=True)
feature_importances.to_csv(fi_file, index=False)
print(f"特征重要性保存到: {fi_file}")

# ===================== 模型保存 =====================
model_file = f"/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/our/SIFT1M/k50/efS500_s10000_nestim100_all_feats_recall_selector_0.90.txt"
os.makedirs(os.path.dirname(model_file), exist_ok=True)
model.booster_.save_model(model_file)

print(f"模型保存到: {model_file}")
print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate} | Saved to: {model_file}")

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# 🔥 使用全部特征
# =========================================================
print("\n===== Start Deep Decision Tree Distillation =====")

all_features = [
    "step", "dists", "inserts","first_nn_dist", "nn_dist", "furthest_dist",
    "avg_dist", "variance","avg_hop","entry_query_dist_ratio",
    "dist_distribution_entropy","cosine_mean","cosine_variance","cosine_direction_entropy"
]

selected_features = all_features
X_small = merged_final[selected_features].values

# =========================================================
# ✅ 划分训练 / 验证
# =========================================================
X_train, X_val, y_train_split, y_val = train_test_split(
    X_small, y_train, test_size=0.2, random_state=42
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
# =========================================================
# ✅ Step 1: 训练基础树
# =========================================================
base_tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42
)
base_tree.fit(X_train, y_train_split)

# =========================================================
# ✅ Step 2: 自动选择 ccp_alpha
# =========================================================
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

print(f"\nBest alpha: {best_alpha}")
print(f"Validation Accuracy: {best_acc:.4f}")

tree = best_tree

# =========================================================
# 🔥 Step 3: 删除“无效分裂”（核心函数）
# =========================================================
def prune_invalid_splits(tree):
    tree_ = tree.tree_

    def is_leaf(node):
        return tree_.children_left[node] == _tree.TREE_LEAF

    # 获取子树最终类别
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

        left_class = get_final_class(left)
        right_class = get_final_class(right)

        # ⭐ 核心：如果 split 没改变结果 → 删除
        if left_class == right_class:
            tree_.children_left[node] = _tree.TREE_LEAF
            tree_.children_right[node] = _tree.TREE_LEAF

    recurse(0)

# 执行剪枝
prune_invalid_splits(tree)

# =========================================================
# ✅ Step 4: 训练精度
# =========================================================
train_acc = tree.score(X_small, y_train)
print(f"\nTree Train Accuracy: {train_acc:.4f}")

# =========================================================
# ✅ Step 5: 输出规则
# =========================================================
rules = clean_export_tree(tree, selected_features)

print("\n===== Clean Decision Tree Rules =====")
print(rules[:2000])

# =========================================================
# ✅ Step 6: 导出 if-else（部署用）
# =========================================================
def tree_to_code(tree, feature_names, max_depth=6):
    tree_ = tree.tree_

    def recurse(node, depth):
        indent = "    " * depth

        if depth > max_depth:
            print(f"{indent}return \"UNKNOWN\"")
            return

        if tree_.feature[node] != -2:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            print(f"{indent}if ({name} <= {threshold:.6f}):")
            recurse(tree_.children_left[node], depth + 1)

            print(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            value = tree_.value[node][0]
            pred = np.argmax(value)
            label = "CNS" if pred == 0 else "RECALL"
            print(f"{indent}return \"{label}\"")

    recurse(0, 0)

print("\n===== Deployable Code =====")
tree_to_code(tree, selected_features)

# =========================================================
# ✅ Step 7: 保存规则
# =========================================================
rule_file = "/home/extra_home/lxx23125236/ali/DARTH-main-data/result/our/SIFT1M/k50/decision_tree_rules_2000.txt"

with open(rule_file, "w") as f:
    f.write(rules)

print(f"\n✅ 决策树规则已保存: {rule_file}")

