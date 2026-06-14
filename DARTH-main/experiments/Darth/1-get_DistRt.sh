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
  "SIFT1M 32 500 500"
  "GLOVE100 16 500 500"
  "GIST1M 32 500 1000"
  "DEEP10M 32 500 750"
)


k_values=(100)
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
#sift
#0.95 distRT：2349  ipi:1174    mpi:235
#0.96 distRT：2545  ipi:1272    mpi:255
#0.97 distRT：2795  ipi:1397    mpi:280
#0.98 distRT：3137  ipi:1568    mpi:314
#0.99 distRT：3646  ipi:1823    mpi:365
#gist
#0.95 distRT：7547  ipi:3774    mpi:755
#0.96 distRT：8205  ipi:4103    mpi:821
#0.97 distRT：9008  ipi:4504    mpi:901
#0.98 distRT：9837  ipi:4919    mpi:984
#0.99 distRT：10746  ipi:5373    mpi:1075
#glove
#0.95 distRT：877  ipi:439    mpi:88
#0.96 distRT：915  ipi:458    mpi:92
#0.97 distRT：964  ipi:482    mpi:96
#0.98 distRT：1030  ipi:515    mpi:103
#0.99 distRT：1135  ipi:568    mpi:114
#deep
#0.95 distRT：3545  ipi:1773    mpi:355
#0.96 distRT：3875  ipi:1938    mpi:388
#0.97 distRT：4307  ipi:2154    mpi:431
#0.98 distRT：4823  ipi:2412    mpi:482
#0.99 distRT：5617  ipi:2809    mpi:562
