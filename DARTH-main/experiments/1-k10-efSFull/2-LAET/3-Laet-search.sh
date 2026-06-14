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

#sample=1000
train_queries=10000
experiment_times=3

mode=laet-early-stop-testing
mkdir -p  ${RESULTS_DIRECTORY}/laet


dataset_params=(
  "SIFT1M 32 500 250 1 10000"
  "GLOVE100 16 500 1000 1 10000"
  "GIST1M 32 500 600 1 1000"
  "DEEP10M 32 500 200 1 10000"
)

k_values=(10)

SIFT1M_tuples_k10=(
  "0.95 250 241 1.06"
  "0.96 250 241 1.11"
  "0.97 250 241 1.21"
  "0.98 250 241 1.36"
  "0.99 250 241 1.61"
)


GLOVE100_tuples_k10=(
  "0.95 1000 200 1.11"
  "0.96 1000 200 1.16"
  "0.97 1000 200 1.36"
  "0.98 1000 200 1.61"
  "0.99 1000 200 2.46"
)

GIST1M_tuples_k10=(
  "0.95 600 1260 1.06"
  "0.96 600 1260 1.11"
  "0.97 600 1260 1.26"
  "0.98 600 1260 1.41"
  "0.99 600 1260 1.86"
)

DEEP10M_tuples_k10=(
  "0.95 200 368 0.81"
  "0.96 200 368 0.96"
  "0.97 200 368 1.16"
  "0.98 200 368 1.36"
  "0.99 200 368 1.71"
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
            if [ "$k" == "10" ]; then
                tuples=("${SIFT1M_tuples_k10[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "10" ]; then
                tuples=("${DEEP10M_tuples_k10[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "10" ]; then
                tuples=("${GIST1M_tuples_k10[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "10" ]; then
                tuples=("${GLOVE100_tuples_k10[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
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

              python /root/DARTH-main/experiments/0-ALL_graphIndex_trainData/多次搜索取平均.py \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/times/efS${efS}_qs${sample}_tr${target_recall}_f${F} \
                  ${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}_f${F}.txt \
                  $experiment_times

              echo "多次搜索中每个查询点的平均结果，写入完成：${RESULTS_DIRECTORY}/laet/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}_f${F}.txt"

              python /root/DARTH-main/experiments/0-ALL_graphIndex_trainData/REM-Laet-早停统计.py \
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
