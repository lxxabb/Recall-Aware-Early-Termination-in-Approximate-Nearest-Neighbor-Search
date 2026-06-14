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
  "SIFT1M 500 10000"
  "GLOVE100 500 10000"
  "GIST1M 1000 10000"
  "DEEP10M 750 10000"
)

mode=our_Raet_recall_get_StablizeCount
k_values=(100)
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

--------------- SIFT1M ---------------
--------------- k=100 ---------------
>>> target_recall=0.95
训练集平均稳定次数为: 61

>>> target_recall=0.96
训练集平均稳定次数为: 65

>>> target_recall=0.97
训练集平均稳定次数为: 59

>>> target_recall=0.98
训练集平均稳定次数为: 51

>>> target_recall=0.99
训练集平均稳定次数为: 24

--------------- GLOVE100 ---------------
--------------- k=100 ---------------
>>> target_recall=0.95
训练集平均稳定次数为: 63

>>> target_recall=0.96
训练集平均稳定次数为: 64

>>> target_recall=0.97
训练集平均稳定次数为: 43

>>> target_recall=0.98
训练集平均稳定次数为: 35

>>> target_recall=0.99
训练集平均稳定次数为: 18

-------------- GIST1M ---------------
--------------- k=100 ---------------
>>> target_recall=0.95
训练集平均稳定次数为: 184

>>> target_recall=0.96
训练集平均稳定次数为: 205

>>> target_recall=0.97
训练集平均稳定次数为: 211

>>> target_recall=0.98
训练集平均稳定次数为: 204

>>> target_recall=0.99
训练集平均稳定次数为: 84

--------------- DEEP10M ---------------
--------------- k=100 ---------------

>>> target_recall=0.95
训练集平均稳定次数为: 108

>>> target_recall=0.96
训练集平均稳定次数为: 119

>>> target_recall=0.97
训练集平均稳定次数为: 122

>>> target_recall=0.98
训练集平均稳定次数为: 116

>>> target_recall=0.99
训练集平均稳定次数为: 45
