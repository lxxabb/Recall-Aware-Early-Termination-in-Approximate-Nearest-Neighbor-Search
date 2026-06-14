#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
export LD_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/LightGBM/lib:$LD_LIBRARY_PATH
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/NSG/cmake-build-lxx-release

#/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
#  -S /home/extra_home/lxx23125236/ali/NSG \
#  -B ${BUILD_DIR} \
#  -G Ninja \
#  -DCMAKE_BUILD_TYPE=Release \
#
## -------- 2️⃣ 编译 --------
#ninja -C ${BUILD_DIR} test_nsg_optimized_search
#
#
#echo ""
#echo "============= Build Done (Release) ============="
#echo ""
#数据集 efS即搜索时的CNS 训练集数量
dataset_params=(
#  "SIFT1M 500 10000"
#  "GLOVE100 500 10000"
#  "GIST1M 1000 10000"
#  "DEEP10M 750 10000"
  "T2I1M 1000 10000"

)
k_values=(50)
TRAINING_DATA_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/train_data
mode=Laet_Darth_train_data
for dataset_param in "${dataset_params[@]}"
do
    read ds efS query_num <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
      mkdir -p ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}
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
                --out_put_path=${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}/efS${efS}_qs${query_num}_full.txt
        echo "Laet_Darth方法，在${ds}训练集 k=${k}收集Laet_Darth特征完毕"
    done
    echo ""
    echo ""
done
