#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/DARTH-main/cmake-build-lxx-release

/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /home/extra_home/lxx23125236/ali/DARTH-main \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBLA_VENDOR=OpenBLAS \
  -DBLAS_LIBRARIES=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib/libopenblas.so \
  -DLAPACK_LIBRARIES=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib/libopenblas.so \
  -DCMAKE_INCLUDE_PATH=/home/extra_home/lxx23125236/ali/OpenBLAS-install/include \
  -DCMAKE_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} faiss
ninja -C ${BUILD_DIR} hnsw_test


echo ""
echo "============= Build Done (Release) ============="
echo ""

#模型选择器的生成训练数据代码，执行中的模型路径无用，但需要填写正确，即需要真正有模型，但不会用到。

INDEX_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/data/HNSW-index
DATASET_DIRECTORY=/home/extra_home/lxx23125236/ann-data/
RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/result
MODEL_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/raet
MODEL_DIRECTORY_CNS=/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/raet-cns-regression_CNSFull
MODEL_DIRECTORY_SELECTOR=/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models/our
#在第一次预测准确率时，收集特征
mode=raet-select-model-data-gen
n_estim=100
train_queries=10000
experiment_times=1
#datasetname M efC efFull li默认为1,模型训练使用的是efFull，但在搜索时使用efS=CNS=刚好达到要求准确率时的最小CNS

dataset_params=(
  "SIFT1M 32 500 500 1 10000"
#  "GLOVE100 16 500 500 1 10000"
#  "GIST1M 32 500 1000 1 1000"
#  "DEEP10M 32 500 750 1 10000"
)

k_values=(100)

#期望准确率 达到要求准确率时的最小CNS 和平均稳定次数参数 平均达到1准确率时的最小稳定次数
SIFT1M_tuples_k100=(
  "0.95 500 60 157 70"
#  "0.96 500 64 153 70"
#  "0.97 500 68 147 70"
#  "0.98 500 71 136 70"
#  "0.99 500 67 110 70"
)

GLOVE100_tuples_k100=(
  "0.95 500 56 106 91"
  "0.96 500 53 98 91"
  "0.97 500 42 81 91"
  "0.98 500 37 68 91"
  "0.99 500 33 55 91"
)

GIST1M_tuples_k100=(
  "0.95 1000 221 421 84"
  "0.96 1000 241 415 84"
  "0.97 1000 265 406 84"
  "0.98 1000 281 387 84"
  "0.99 1000 267 332 84"
)

DEEP10M_tuples_k100=(
  "0.95 750 82 202 60"
  "0.96 750 89 198 60"
  "0.97 750 98 193 60"
  "0.98 750 105 182 60"
  "0.99 750 105 155 60"
)

mkdir -p  ${RESULTS_DIRECTORY}/our

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFull li sample<<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir -p  ${RESULTS_DIRECTORY}/our/${ds}

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        if [ "$ds" == "SIFT1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${SIFT1M_tuples_k10[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GIST1M_tuples_k10[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GLOVE100_tuples_k10[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${DEEP10M_tuples_k10[@]}")
            fi
        fi

        mkdir -p  ${RESULTS_DIRECTORY}/our/${ds}/k${k}
        mkdir -p  ${RESULTS_DIRECTORY}/our/${ds}/k${k}/train-data/

        for tuple in "${tuples[@]}"; do
          read target_recall efS stability_times stability_times_r1 quantile<<< "$tuple"

          echo ">>> target_recall=${target_recall}, efS=${efS}"

          if true; then
            for time in $(seq 1 $experiment_times)
            do

                ${BUILD_DIR}/hnsw-test/hnsw_test \
                  --dataset ${ds} \
                  --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                  --query-num ${sample} --k $k \
                  --output $RESULTS_DIRECTORY/our/${ds}/k${k}/train-data/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --mode ${mode} \
                  --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                  --dataset-dir-prefix ${DATASET_DIRECTORY} \
                  --target-recall ${target_recall} \
                  --stability-times ${stability_times} \
                  --stability-times-r1 ${stability_times_r1} \
                  --query-type training \
                  --predictor-model-path ${MODEL_DIRECTORY}/${ds}/k${k}/efS${efSFull}_s${train_queries}_nestim${n_estim}_all_feats_Noquery_duration.txt \
                  --predictor-model-path-select-CNS ${MODEL_DIRECTORY_CNS}/${ds}/k${k}/efS${efS}_s${train_queries}_nestim${n_estim}_all_feats_${target_recall}_CNSFull_quantile${quantile}_Noquery_Noske_Nodensty.txt \
                  --model-selector-path ${MODEL_DIRECTORY_SELECTOR}/${ds}/k${k}/efS${efS}_s${train_queries}_nestim${n_estim}_all_feats_${target_recall}.txt
            done
          fi
        done
    done
    echo ""
    echo ""
done