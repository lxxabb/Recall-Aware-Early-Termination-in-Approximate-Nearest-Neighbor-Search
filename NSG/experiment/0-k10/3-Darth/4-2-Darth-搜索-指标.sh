#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/root/NSG/cmake-build-lxx-release

/root/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /root/NSG \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} test_nsg_optimized_search


echo ""
echo "============= Build Done (Release) ============="
echo ""
#数据集和每个数据集中  构建训练数据集时采用的efS 测试集中的查询点数量
dataset_params=(
  "SIFT1M 500 10000"
#  "GLOVE100 500 10000"
#  "GIST1M 1000 1000"
#  "DEEP10M 750 10000"
)
MODEL_DIRECTORY=/root/NSG-data/predictor_models/darth
RESULTS_DIRECTORY=/root/NSG-data/results/darth
experiment_times=1
mode=Darth_test_metrics
k_values=(100)
train_queries=10000
n_estimators=100
#要求准确率，搜索时的CNS大小，ipi，mpi
SIFT1M_tuples_k10=(
  "0.95 32 2200 265"
  "0.96 36 2726 140"
  "0.97 43 3400 306"
  "0.98 54 4100 336"
  "0.99 77 5000 381"
)

GIST1M_tuples_k10=(
  "0.95 116 9000 777"
  "0.96 136 10000 840"
  "0.97 166 12000 918"
  "0.98 214 15000 1013"
  "0.99 346 21000 1133"
)

DEEP10M_tuples_k10=(
  "0.95 23 3700 413"
  "0.96 31 4200 442"
  "0.97 45 5500 477"
  "0.98 72 8500 518"
  "0.99 145 13000 573"
)

for dataset_param in "${dataset_params[@]}"
do
    read ds efS_train query_num<<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

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
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${DEEP10M_tuples_k10[@]}")
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
        fi

        for tuple in "${tuples[@]}"; do
          read target_recall efS ipi mpi <<< "$tuple"
          echo ">>> target_recall=${target_recall}"
          mkdir -p ${RESULTS_DIRECTORY}/${ds}/k${k}
          ${BUILD_DIR}/tests/test_nsg_optimized_search \
            --dataset=${ds} \
            --K=$k \
            --L=${efS} \
            --mode=${mode} \
            --query_type=test \
            --query_num=${query_num} \
            --ipi=${ipi} \
            --mpi=${mpi} \
            --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_nestim${n_estimators}_all_feats.txt \
            --target_recall=${target_recall} \
            --out_put_path=${RESULTS_DIRECTORY}/${ds}/k${k}/metrics_recall${target_recall}.json
        done
    done
    echo ""
    echo ""
done
