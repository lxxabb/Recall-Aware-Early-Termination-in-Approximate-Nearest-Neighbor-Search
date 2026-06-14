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
#数据集 efS即搜索时的CNS 训练集数量
dataset_params=(
  "SIFT1M 500 10000"
  "GLOVE100 500 10000"
  "GIST1M 1000 10000"
  "DEEP10M 750 10000"
)
k_values=(100)
TRAINING_DATA_DIRECTORY=/root/NSG-data/train_data/raet_CNS
mode=our_Raet_CNS_train_label
SIFT1M_tuples_k100=(
  "0.95 100"
  "0.96 100"
  "0.97 106"
  "0.98 128"
  "0.99 174"
)


GLOVE100_tuples_k100=(
  "0.95 100"
  "0.96 100"
  "0.97 100"
  "0.98 100"
  "0.99 210"
)

GIST1M_tuples_k100=(
  "0.95 220"
  "0.96 255"
  "0.97 306"
  "0.98 394"
  "0.99 591"
)

DEEP10M_tuples_k100=(
  "0.95 123"
  "0.96 148"
  "0.97 188"
  "0.98 257"
  "0.99 432"
)
for dataset_param in "${dataset_params[@]}"
do
    read ds efSFull query_num <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
      mkdir -p ${TRAINING_DATA_DIRECTORY}/${ds}/k${k}/label
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
          read target_recall efS <<< "$tuple"
          echo ">>> target_recall=${target_recall}"
            ${BUILD_DIR}/tests/test_nsg_optimized_search \
                --dataset=${ds} \
                --K=$k \
                --L=${efS} \
                --mode=${mode} \
                --query_type=train \
                --query_num=${query_num} \
                --target_recall=${target_recall} \
                --out_put_path=${TRAINING_DATA_DIRECTORY}/${ds}/k${k}/label/efS${efSFull}_qs${query_num}_${target_recall}.txt
          echo "Raet_CNS方法，在${ds}训练集 k=${k}收集Raet_CNS标签完毕"
        done
    done
    echo ""
    echo ""
done
