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

TRAIN_DATA_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/train_data/Laet_Darth_train_data/

sample=10000
experiment_times=3
dataset_params=(
#  "SIFT1M  500"
#  "GLOVE100  500"
#  "GIST1M  1000"
#  "DEEP10M  750"
  "T2I1M 1000"
)


k_values=(50)
recall_values=(0.90 0.92 0.94 0.96 0.98)

for dataset_param in "${dataset_params[@]}"
do
    read ds efSFULL <<< "$dataset_param"
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""
    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
        for recall in "${recall_values[@]}"; do
          if true; then
            python /home/extra_home/lxx23125236/ali/NSG/experiment/0-k50-full/3-Darth/1-1-获取darth需要的distRT.py \
                  "${TRAIN_DATA_DIRECTORY}${ds}/k${k}/efS${efSFULL}_qs${sample}_full.txt" \
                  "$recall"
          fi
        done
    done
    echo ""
    echo ""
done
exit

