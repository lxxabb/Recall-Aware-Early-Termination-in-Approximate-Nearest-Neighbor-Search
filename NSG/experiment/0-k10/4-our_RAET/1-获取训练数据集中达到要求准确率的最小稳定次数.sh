#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/root/NSG/cmake-build-lxx-release

/root/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /root/NSG \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} test_nsg_optimized_search

echo ""
echo "============= Build Done (Release) ============="
echo ""

dataset_params=(
  "SIFT1M 150 10000"
  "GIST1M 600 10000"
  "DEEP10M 300 10000"
)

mode=our_Raet_recall_get_StablizeCount
k_values=(10)
train_queries=10000
recall_values=(0.95 0.96 0.97 0.98 0.99)

for dataset_param in "${dataset_params[@]}"
do
    read ds efS_train query_num<<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
        for target_recall in "${recall_values[@]}"; do
          echo ">>> target_recall=${target_recall}"

            ${BUILD_DIR}/tests/test_nsg_optimized_search \
              --dataset=${ds} \
              --K=$k \
              --L=${efS_train} \
              --mode=${mode} \
              --query_type=train \
              --query_num=${query_num} \
              --target_recall=${target_recall}
        done
    done
    echo ""
    echo ""
done
