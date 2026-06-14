#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/DARTH-main/cmake-build-lxx-release

#/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
#  -S /home/extra_home/lxx23125236/ali/DARTH-main \
#  -B ${BUILD_DIR} \
#  -G Ninja \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DFAISS_ENABLE_GPU=OFF \
#  -DBUILD_SHARED_LIBS=ON \
#  -DFAISS_ENABLE_PYTHON=OFF \
#  -DBUILD_TESTING=OFF \
#  -DFAISS_OPT_LEVEL=avx2 \
#  -DBLA_VENDOR=OpenBLAS \
#  -DBLAS_LIBRARIES=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib/libopenblas.so \
#  -DLAPACK_LIBRARIES=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib/libopenblas.so \
#  -DCMAKE_INCLUDE_PATH=/home/extra_home/lxx23125236/ali/OpenBLAS-install/include \
#  -DCMAKE_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/OpenBLAS-install/lib
#
## -------- 2️⃣ 编译 --------
#ninja -C ${BUILD_DIR} faiss
#ninja -C ${BUILD_DIR} hnsw_test
#
#
#echo ""
#echo "============= Build Done (Release) ============="
#echo ""

INDEX_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/data/HNSW-index
DATASET_DIRECTORY=/home/extra_home/lxx23125236/ann-data/
RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/result
MODEL_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/predictor_models

#sample=1000
train_queries=10000
experiment_times=3

mode=laet-early-stop-testing
mkdir -p  ${RESULTS_DIRECTORY}/laet


dataset_params=(
#  "SIFT1M 32 500 500 1 10000"
#  "GLOVE100 16 500 500 1 10000"
#  "GIST1M 32 500 1000 1 1000"
#  "DEEP10M 32 500 750 1 10000"
  "T2I1M 80 1000 1000 1 10000"
)

k_values=(50)

SIFT1M_tuples_k50=(
  "0.90 500 241 0.46"
  "0.92 500 241 0.51"
  "0.94 500 241 0.61"
  "0.96 500 241 0.71"
  "0.98 500 241 0.91"
)


GLOVE100_tuples_k50=(
  "0.90 500 200 0.61"
  "0.92 500 200 0.66"
  "0.94 500 200 0.76"
  "0.96 500 200 0.81"
  "0.98 500 200 1.01"
)

GIST1M_tuples_k50=(
  "0.90 1000 1260 0.41"
  "0.92 1000 1260 0.46"
  "0.94 1000 1260 0.56"
  "0.96 1000 1260 0.66"
  "0.98 1000 1260 0.91"
)

DEEP10M_tuples_k50=(
  "0.90 750 368 0.41"
  "0.92 750 368 0.46"
  "0.94 750 368 0.51"
  "0.96 750 368 0.66"
  "0.98 750 368 0.86"
)
T2I1M_tuples_k50=(
  "0.90 1000 400 0.46"
  "0.92 1000 400 0.51"
  "0.94 1000 400 0.61"
  "0.96 1000 400 0.76"
  "0.98 1000 400 0.96"
)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFull li sample<<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir -p  ${RESULTS_DIRECTORY}/laet/${ds}

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        mkdir -p  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}
        mkdir -p  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/times/


        if [ "$ds" == "SIFT1M" ]; then
            if [ "$k" == "50" ]; then
                tuples=("${SIFT1M_tuples_k50[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "50" ]; then
                tuples=("${DEEP10M_tuples_k50[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "50" ]; then
                tuples=("${GIST1M_tuples_k50[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "50" ]; then
                tuples=("${GLOVE100_tuples_k50[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            fi
        elif [ "$ds" == "T2I1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${T2I1M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${T2I1M_tuples_k50[@]}")
            fi
        fi

        for tuple in "${tuples[@]}"; do
            read target_recall efS F multiplier <<< "$tuple"
            echo ">>> target_recall=${target_recall}, efS=${efS}"

            JSON_FILE=${RESULTS_DIRECTORY}/laet/${ds}/k${k}/times_summary_tr${target_recall}.json
            echo "[]" > "$JSON_FILE"

            if true; then
              for time in $(seq 1 $experiment_times)
              do
                out="$(
                  ${BUILD_DIR}/hnsw-test/hnsw_test \
                          --dataset ${ds} \
                          --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                          --query-num ${sample} --k $k \
                          --output ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/times/efS${efS}_qs${sample}_tr${target_recall}_f${F}_t${time}.txt \
                          --mode ${mode} \
                          --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                          --dataset-dir-prefix ${DATASET_DIRECTORY} \
                          --target-recall ${target_recall} \
                          --fixed-amount-of-search ${F} --prediction-multiplier ${multiplier} \
                          --query-type testing \
                          --predictor-model-path ${MODEL_DIRECTORY}/laet-our/${ds}/k${k}/efS${efSFull}_s${train_queries}_f${F}.txt
                )"
                last_line=$(echo "$out" | tail -n 1)
                echo "最后一行日志: $last_line"

                # ================== 解析字段 ==================
                M_val=$(echo "$last_line" | sed -n 's/.*Index\[M=\([0-9]*\).*/\1/p')
                efC_val=$(echo "$last_line" | sed -n 's/.*efC=\([0-9]*\).*/\1/p')
                efS_val=$(echo "$last_line" | sed -n 's/.*efS=\([0-9]*\)\].*/\1/p')
                index_time=$(echo "$last_line" | sed -n 's/.*IndexTime: \([0-9.]*\)s.*/\1/p')
                search_time=$(echo "$last_line" | sed -n 's/.*SearchTime: \([0-9.]*\)s.*/\1/p')
                total_time=$(echo "$last_line" | sed -n 's/.*TotalTime: \([0-9.]*\)s.*/\1/p')
                recall=$(echo "$last_line" | sed -n 's/.*Recall@[0-9]*: \([0-9.]*\).*/\1/p')

                # ================== 追加到 JSON ==================
                tmp=$(mktemp)
                jq \
                  --arg time "$time" \
                  --arg tr "$target_recall" \
                  --arg M "$M_val" \
                  --arg efC "$efC_val" \
                  --arg efS "$efS_val" \
                  --arg index_time "$index_time" \
                  --arg search_time "$search_time" \
                  --arg total_time "$total_time" \
                  --arg recall "$recall" \
                  '. += [{
                    time: ($time|tonumber),
                    target_recall: ($tr|tonumber),
                    M: ($M|tonumber),
                    efC: ($efC|tonumber),
                    efS: ($efS|tonumber),
                    index_time: ($index_time|tonumber),
                    search_time: ($search_time|tonumber),
                    total_time: ($total_time|tonumber),
                    recall: ($recall|tonumber)
                  }]' \
                  "$JSON_FILE" > "$tmp" && mv "$tmp" "$JSON_FILE"
              done

              # ================== 计算 search_time 在所有time中的均值并追加一条 ==================
              tmp=$(mktemp)
              jq '
                . as $all
                | ($all | map(.search_time) | add / length) as $mean
                | . += [{
                    summary: "mean_search_time",
                    value: $mean
                  }]
              ' "$JSON_FILE" > "$tmp" && mv "$tmp" "$JSON_FILE"


              echo "多次搜索总的时间结果，写入完成：$JSON_FILE"

              python /home/extra_home/lxx23125236/ali/DARTH-main/experiments/0-ALL_graphIndex_trainData/多次搜索取平均.py \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/times/efS${efS}_qs${sample}_tr${target_recall}_f${F} \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}_f${F}.txt \
                  $experiment_times

              echo "多次搜索中每个查询点的平均结果，写入完成：${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}_f${F}.txt"

              python /home/extra_home/lxx23125236/ali/DARTH-main/experiments/0-ALL_graphIndex_trainData/REM-Laet-早停统计.py \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}_f${F}.txt \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_statistics_tr${target_recall}.json \
                  $target_recall

              echo "从平均搜索结果文件中，所有查询点的早停统计结果statistics，写入完成：${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_statistics_tr${target_recall}.json"

            fi

        done

    done

    echo ""
    echo ""
done
