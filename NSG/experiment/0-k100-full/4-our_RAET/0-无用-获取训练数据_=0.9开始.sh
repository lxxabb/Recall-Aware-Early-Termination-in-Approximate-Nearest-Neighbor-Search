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
TRAINING_DATA_DIRECTORY=/root/NSG-data/train_data/raet
mode=our_Raet_train_data
for dataset_param in "${dataset_params[@]}"
do
    read ds efS query_num <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
      mkdir -p ${TRAINING_DATA_DIRECTORY}/${ds}/k${k}
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
            ${BUILD_DIR}/tests/test_nsg_optimized_search \
                --dataset=${ds} \
                --K=$k \
                --L=${efS} \
                --mode=${mode} \
                --query_type=train \
                --query_num=${query_num} \
                --out_put_path=${TRAINING_DATA_DIRECTORY}/${ds}/k${k}/efS${efS}_qs${query_num}.txt
        echo "Raet方法，在${ds}训练集 k=${k}收集raet特征完毕"
    done
    echo ""
    echo ""
done
