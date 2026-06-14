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
RESULTS_DIRECTORY=/root/DARTH-main-data/result
MODEL_DIRECTORY=/root/DARTH-main-data/predictor_models

sample=10000
train_queries=10000
experiment_times=1

mode=raet-get-stability-times-testing

dataset_params=(
  "SIFT1M 32 500 500 1"
  "GLOVE100 16 500 500 1"
  "GIST1M 32 500 1000 1"
  "DEEP10M 32 500 750 1"
)

k_values=(100)

SIFT1M_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)


GLOVE100_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)

GIST1M_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)

DEEP10M_tuples_k100=(
  "0.95"
  "0.96"
  "0.97"
  "0.98"
  "0.99"
)


for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

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
        mkdir -p  ${RESULTS_DIRECTORY}/raet/${ds}/k${k}/stablity/
        for tuple in "${tuples[@]}"; do
            read target_recall<<< "$tuple"
            echo ">>> 模型随意选择，不会被调用，target_recall=${target_recall}, efS=${efS}"

            if true; then
              for time in $(seq 1 $experiment_times)
              do
                ${BUILD_DIR}/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${sample} --k $k \
                        --mode ${mode} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --target-recall ${target_recall} \
                        --query-type training \
                        --stability_log_path $RESULTS_DIRECTORY/raet/${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                        --predictor-model-path ${MODEL_DIRECTORY}/darth-our/${ds}/k${k}/efS${efS}_s${train_queries}_nestim100_all_feats.txt
              done
            fi
            echo "=========================================="

        done

    done

    echo ""
    echo ""
done
