#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/root/DARTH-main/cmake-build-lxx-release

/root/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /root/DARTH-main \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBLA_VENDOR=OpenBLAS \
  -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so \
  -DLAPACK_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so \
  -DCMAKE_INCLUDE_PATH=/opt/OpenBLAS/include \
  -DCMAKE_LIBRARY_PATH=/opt/OpenBLAS/lib

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} faiss
ninja -C ${BUILD_DIR} hnsw_test


echo ""
echo "============= Build Done (Release) ============="
echo ""

TRAIN_DATA_DIRECTORY=/root/NSG-data/train_data/Laet_Darth_train_data/

sample=10000
experiment_times=3
dataset_params=(
  "SIFT1M  300"
)


k_values=(100)
recall_values=(0.95 0.96 0.97 0.98 0.99)

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
            python /root/NSG/experiment/0-k100-full/3-Darth/1-1-获取darth需要的distRT.py \
                  "${TRAIN_DATA_DIRECTORY}${ds}/k${k}/efS${efSFULL}_qs${sample}.txt" \
                  "$recall"
          fi
        done
    done
    echo ""
    echo ""
done
exit
--------------- SIFT1M ---------------


--------------- k=100 ---------------

目标准确率: r >= 0.9500
成功达到目标的查询数量: 9976
平均距离计算次数 (dists): 2440.79
ipi: 1220
mpi: 244
===============================
目标准确率: r >= 0.9600
成功达到目标的查询数量: 9949
平均距离计算次数 (dists): 2603.76
ipi: 1302
mpi: 260
===============================
目标准确率: r >= 0.9700
成功达到目标的查询数量: 9898
平均距离计算次数 (dists): 2800.30
ipi: 1400
mpi: 280
===============================
目标准确率: r >= 0.9800
成功达到目标的查询数量: 9761
平均距离计算次数 (dists): 3037.00
ipi: 1519
mpi: 304
===============================
目标准确率: r >= 0.9900
成功达到目标的查询数量: 9430
平均距离计算次数 (dists): 3342.98
ipi: 1671
mpi: 334
===============================