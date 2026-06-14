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
#  "SIFT1M 10000"
#  "GLOVE100 10000"
#  "GIST1M 10000"
  "DEEP10M 10000"
)


k_values=(100)


SIFT1M_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)


GLOVE100_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)


GIST1M_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)

DEEP10M_tuples_k100=(
#  "0.95"
#  "0.96"
  "0.97"
#  "0.98"
#  "0.99"
)


mode=get_train_CNS
for dataset_param in "${dataset_params[@]}"
do
    read ds query_num <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        if [ "$ds" == "SIFT1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${SIFT1M_tuples_k10[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${DEEP10M_tuples_k10[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GIST1M_tuples_k10[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GLOVE100_tuples_k10[@]}")
            fi
        fi

        for tuple in "${tuples[@]}"; do
          read target_recall<<< "$tuple"
          echo ">>> target_recall=${target_recall}"
              ${BUILD_DIR}/tests/test_nsg_optimized_search \
                  --dataset=${ds} \
                  --K=$k \
                  --mode=${mode} \
                  --query_type=train \
                  --query-num=${query_num} \
                  --target_recall=${target_recall}
          echo "数据集${ds}达到要求准确率${target_recall}的最小CNS已找到"
        done
    done
    echo ""
    echo ""
done
