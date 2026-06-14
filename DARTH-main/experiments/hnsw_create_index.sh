#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/root/DARTH-main/cmake-build-lxx-release

cmake -S . -B ${BUILD_DIR} \
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


# -------- 3️⃣ 运行实验 --------

dataset_params=(
#    "SIFT1M 32 500 500"
    "GIST1M 32 500 1000"
#    "GLOVE100 16 500 500"
    "DEEP10M 32 500 750"
)

INDEX_DIRECTORY=/root/DARTH-main-data/data/HNSW-index
DATASET_DIRECTORY=/root/ann-data/

mode=no-early-stop

# ⚠️ 之前脚本里用到了 sample 和 query_type，但没有定义，我补上默认值
sample=10000
query_type=0

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"

    ./${BUILD_DIR}/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k 100 \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
        --query-type ${query_type} \
        --dataset-dir-prefix ${DATASET_DIRECTORY} \
        --save-index
done
