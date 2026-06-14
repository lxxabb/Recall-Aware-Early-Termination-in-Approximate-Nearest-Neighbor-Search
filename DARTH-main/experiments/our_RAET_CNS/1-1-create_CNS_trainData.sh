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

dataset_params=(
    "SIFT1M 32 500 500"
    "GLOVE100 16 500 500"
    "GIST1M 32 500 1000"
    "DEEP10M 32 500 750"
)

k_values=(100)

TRAINING_DATA_DIRECTORY=/root/DARTH-main-data/data/et_training_data
INDEX_DIRECTORY=/root/DARTH-main-data/data/HNSW-index
DATASET_DIRECTORY=/root/ann-data/

echo ""
echo "============= 生成训练数据集 (Release) ============="
echo ""

mode=raet-CNS-fea-training
mkdir -p ${TRAINING_DATA_DIRECTORY}
mkdir -p ${TRAINING_DATA_DIRECTORY}/${mode}
for query_num in 100000
do
    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS <<< "$dataset_param"

        mkdir -p ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}

        echo ""
        echo "--------------- ${ds} ---------------"
        echo ""

        for k in "${k_values[@]}"
        do
            mkdir -p ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}

            for logging_interval in 1
            do
                ${BUILD_DIR}/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${query_num} --k $k \
                        --mode ${mode} \
                        --logging-interval ${logging_interval} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --query-type training \
                        --output ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}/efS${efS}_qs${query_num}.txt
            done
        done

        echo ""
        echo ""
    done
done

echo ""
echo "============= CNS训练数据集已生成 (Release) ============="
echo ""
exit

