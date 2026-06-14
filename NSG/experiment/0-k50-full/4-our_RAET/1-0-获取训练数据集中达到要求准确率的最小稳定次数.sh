#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
export LD_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/LightGBM/lib:$LD_LIBRARY_PATH
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/NSG/cmake-build-lxx-release

/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /home/extra_home/lxx23125236/ali/NSG \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} test_nsg_optimized_search


echo ""
echo "============= Build Done (Release) ============="
echo ""

dataset_params=(
#  "SIFT1M 500 10000"
#  "GLOVE100 500 10000"
#  "GIST1M 1000 10000"
#  "DEEP10M 750 10000"
  "T2I1M 1000 10000"
)

mode=our_Raet_recall_get_StablizeCount
RESULT_DATA_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/results/raet
k_values=(50)
train_queries=10000
recall_values=(0.90 0.92 0.94 0.96 0.98)

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
        mkdir -p  ${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/
        for target_recall in "${recall_values[@]}"; do
          echo ">>> target_recall=${target_recall}"

            ${BUILD_DIR}/tests/test_nsg_optimized_search \
              --dataset=${ds} \
              --K=$k \
              --L=${efS_train} \
              --mode=${mode} \
              --query_type=train \
              --query_num=${query_num} \
              --out_put_path=${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/efS${efS_train}_qs${query_num}_tr${target_recall}.txt \
              --target_recall=${target_recall}
        done
    done
    echo ""
    echo ""
done
