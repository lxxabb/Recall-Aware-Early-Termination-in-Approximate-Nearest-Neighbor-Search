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
MODEL_DIRECTORY=/root/DARTH-main-data/predictor_models/raet-cns-regression_CNSFull
mode=raet-CNS-early-stop-testing
n_estim=100
#n_estim=200
train_queries=10000
experiment_times=3
#datasetname M efC efFull li默认为1,模型训练使用的是efFull，但在搜索时使用efS=CNS=刚好达到要求准确率时的最小CNS

dataset_params=(
  "SIFT1M 32 500 500 1 10000"
  "GLOVE100 16 500 500 1 10000"
  "GIST1M 32 500 1000 1 1000"
#  "DEEP10M 32 500 750 1 10000"
)

k_values=(50)

#期望准确率 达到要求准确率时的最小CNS 和模型分位数

SIFT1M_tuples_k50=(
  "0.90 500 72"
  "0.92 500 72"
  "0.94 500 72"
  "0.96 500 72"
  "0.98 500 72"
)

GLOVE100_tuples_k50=(
  "0.90 500 77"
  "0.92 500 77"
  "0.94 500 77"
  "0.96 500 77"
  "0.98 500 77"
)

GIST1M_tuples_k50=(
  "0.90 1000 78"
  "0.92 1000 78"
  "0.94 1000 78"
  "0.96 1000 78"
  "0.98 1000 78"
)

DEEP10M_tuples_k50=(
  "0.90 750 72"
  "0.92 750 72"
  "0.94 750 72"
  "0.96 750 72"
  "0.98 750 72"
)

mkdir -p  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFull li sample<<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir -p  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}

    for k in "${k_values[@]}"
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        if [ "$ds" == "SIFT1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${SIFT1M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${SIFT1M_tuples_k50[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GIST1M_tuples_k50[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GLOVE100_tuples_k50[@]}")
            fi
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${DEEP10M_tuples_k50[@]}")
            fi
        fi

        mkdir -p  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}
        mkdir -p  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/times/

        for tuple in "${tuples[@]}"; do
          read target_recall efS quantile<<< "$tuple"

          echo ">>> target_recall=${target_recall}, efS=${efS}"

          JSON_FILE=${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_times_summary_tr${target_recall}.json
          echo "[]" > "$JSON_FILE"

          if true; then
            for time in $(seq 1 $experiment_times)
            do
              out="$(
                ${BUILD_DIR}/hnsw-test/hnsw_test \
                  --dataset ${ds} \
                  --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                  --query-num ${sample} --k $k \
                  --output $RESULTS_DIRECTORY/raet-cns-regression_CNSFull/${ds}/k${k}/times/efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \
                  --mode ${mode} \
                  --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                  --dataset-dir-prefix ${DATASET_DIRECTORY} \
                  --target-recall ${target_recall} \
                  --query-type testing \
                  --predictor-model-path ${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS}_s${train_queries}_nestim${n_estim}_all_feats_${target_recall}_CNSFull_quantile${quantile}.txt
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
                  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/times/efS${efS}_qs${sample}_tr${target_recall} \
                  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  $experiment_times

            echo "多次搜索中每个查询点的平均结果，写入完成：${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}.txt"

            python /root/DARTH-main/experiments/0-ALL_graphIndex_trainData/Raet-CNS-早停统计.py \
                  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  ${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_statistics_tr${target_recall}.json \
                  $target_recall

            echo "从平均搜索结果文件中，所有查询点的早停统计结果statistics，写入完成：${RESULTS_DIRECTORY}/raet-cns-regression_CNSFull/${ds}/k${k}/efS${efS}_statistics_tr${target_recall}.json"

          fi
        done
    done
    echo ""
    echo ""
done