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

INDEX_DIRECTORY=/root/DARTH-main-data/data/HNSW-index
DATASET_DIRECTORY=/root/ann-data/
MIN_EF_PRE_QUERY_DIRECTORY=/root/DARTH-main-data/data/et_training_data/raet-CNS-fea-training/
MODEL_DIRECTORY=/root/DARTH-main-data/predictor_models

experiment_times=1

mode=optimal-efSearch-per-query

dataset_params=(
  "SIFT1M 32 500 500 1"
  "GLOVE100 16 500 500 1"
  "GIST1M 32 500 1000 1"
  "DEEP10M 32 500 750 1"
)
sample=100000
k_values=(100)

SIFT1M_tuples_k100=(
  "0.95 500"
  "0.96 500"
  "0.97 500"
  "0.98 500"
  "0.99 500"
)


GLOVE100_tuples_k100=(
  "0.95 500"
  "0.96 500"
  "0.97 500"
  "0.98 500"
  "0.99 500"
)

GIST1M_tuples_k100=(
  "0.95 1000"
  "0.96 1000"
  "0.97 1000"
  "0.98 1000"
  "0.99 1000"
)

DEEP10M_tuples_k100=(
  "0.95 750"
  "0.96 750"
  "0.97 750"
  "0.98 750"
  "0.99 750"
)


for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFull li <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        mkdir -p  ${MIN_EF_PRE_QUERY_DIRECTORY}${ds}/k${k}/test


        if [ "$ds" == "SIFT1M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${SIFT1M_tuples_k1[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${DEEP10M_tuples_k1[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${GIST1M_tuples_k1[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${GLOVE100_tuples_k1[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            fi
        fi

        for tuple in "${tuples[@]}"; do
            read target_recall train_CNS<<< "$tuple"
            echo ">>> target_recall=${target_recall}, efS=${efSFull}"

            if true; then
              for time in $(seq 1 $experiment_times)
              do
                  ${BUILD_DIR}/hnsw-test/hnsw_test \
                          --dataset ${ds} \
                          --M ${M} --efConstruction ${efC} --efSearch ${efSFull} \
                          --query-num ${sample} --k $k \
                          --output ${MIN_EF_PRE_QUERY_DIRECTORY}${ds}/k${k}/efs${efSFull}_min_ef_per_query_s${sample}_${target_recall}.csv \
                          --mode ${mode} \
                          --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                          --dataset-dir-prefix ${DATASET_DIRECTORY} \
                          --target-recall ${target_recall} \
                          --query-type training \
                          --train-CNS ${train_CNS}
              done
            fi
        done
    done
    echo ""
    echo ""
done
