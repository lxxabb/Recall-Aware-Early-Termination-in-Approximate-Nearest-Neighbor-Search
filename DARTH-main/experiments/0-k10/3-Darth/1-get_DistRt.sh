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

TRAIN_DATA_DIRECTORY=/root/DARTH-main-data/data/et_training_data/early-stop-training/

sample=10000
experiment_times=3
dataset_params=(
  "SIFT1M 32 500 100"
#  "GLOVE100 16 500 500"
  "GIST1M 32 500 600"
  "DEEP10M 32 500 100"
)


k_values=(10)
recall_values=(0.95 0.96 0.97 0.98 0.99)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFULL <<< "$dataset_param"
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
            python /root/DARTH-main/notebooks_scripts/test/Darth-获取Darth需要的distsRt.py \
                  "${TRAIN_DATA_DIRECTORY}${ds}/k${k}/efS${efSFULL}_qs${sample}.txt" \
                  "$recall"
          fi
        done
    done
    echo ""
    echo ""
done
exit
#--------------- SIFT1M ---------------
#
#
#--------------- k=10 ---------------
#
#目标准确率: r >= 0.9500
#成功达到目标的查询数量: 9471
#平均距离计算次数 (dists): 940.60
#ipi: 470
#mpi: 94
#===============================
#目标准确率: r >= 0.9600
#成功达到目标的查询数量: 9471
#平均距离计算次数 (dists): 940.60
#ipi: 470
#mpi: 94
#===============================
#目标准确率: r >= 0.9700
#成功达到目标的查询数量: 9471
#平均距离计算次数 (dists): 940.60
#ipi: 470
#mpi: 94
#===============================
#目标准确率: r >= 0.9800
#成功达到目标的查询数量: 9471
#平均距离计算次数 (dists): 940.60
#ipi: 470
#mpi: 94
#===============================
#目标准确率: r >= 0.9900
#成功达到目标的查询数量: 9471
#平均距离计算次数 (dists): 940.60
#ipi: 470
#mpi: 94
#===============================
#
#
#
#--------------- GIST1M ---------------
#
#
#--------------- k=10 ---------------
#
#目标准确率: r >= 0.9500
#成功达到目标的查询数量: 9449
#平均距离计算次数 (dists): 3904.53
#ipi: 1952
#mpi: 390
#===============================
#目标准确率: r >= 0.9600
#成功达到目标的查询数量: 9449
#平均距离计算次数 (dists): 3904.53
#ipi: 1952
#mpi: 390
#===============================
#目标准确率: r >= 0.9700
#成功达到目标的查询数量: 9449
#平均距离计算次数 (dists): 3904.53
#ipi: 1952
#mpi: 390
#===============================
#目标准确率: r >= 0.9800
#成功达到目标的查询数量: 9449
#平均距离计算次数 (dists): 3904.53
#ipi: 1952
#mpi: 390
#===============================
#目标准确率: r >= 0.9900
#成功达到目标的查询数量: 9449
#平均距离计算次数 (dists): 3904.53
#ipi: 1952
#mpi: 390
#===============================
#
#
#
#--------------- DEEP10M ---------------
#
#
#--------------- k=10 ---------------
#
#目标准确率: r >= 0.9500
#成功达到目标的查询数量: 9523
#平均距离计算次数 (dists): 718.01
#ipi: 359
#mpi: 72
#===============================
#目标准确率: r >= 0.9600
#成功达到目标的查询数量: 9523
#平均距离计算次数 (dists): 718.01
#ipi: 359
#mpi: 72
#===============================
#目标准确率: r >= 0.9700
#成功达到目标的查询数量: 9523
#平均距离计算次数 (dists): 718.01
#ipi: 359
#mpi: 72
#===============================
#目标准确率: r >= 0.9800
#成功达到目标的查询数量: 9523
#平均距离计算次数 (dists): 718.01
#ipi: 359
#mpi: 72
#===============================
#目标准确率: r >= 0.9900
#成功达到目标的查询数量: 9523
#平均距离计算次数 (dists): 718.01
#ipi: 359
#mpi: 72
#===============================

