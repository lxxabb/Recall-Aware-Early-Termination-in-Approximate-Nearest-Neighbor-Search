#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
export LD_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/LightGBM/lib:$LD_LIBRARY_PATH
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/NSG/cmake-build-lxx-release

/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
  -S /home/extra_home/lxx23125236/ali/NSG \
  -B ${BUILD_DIR} \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \

# -------- 2️⃣ 编译 --------
ninja -C ${BUILD_DIR} test_nsg_optimized_search


echo ""
echo "============= Build Done (Release) ============="
echo ""
#数据集和每个数据集中  构建训练数据集时采用的efS 测试集中的查询点数量
#dataset_params=(
#  "SIFT1M 500 10000"
#  "GLOVE100 500 10000"
#  "GIST1M 1000 10000"
#  "DEEP10M 750 10000"
#  "T2I1M 1000 10000"
#)
dataset_params=(
#  "SIFT1M 500 20000"
  "GLOVE100 500 20000"
  "GIST1M 1000 20000"
  "DEEP10M 750 20000"
  "T2I1M 1000 20000"
)
MODEL_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/predictor_models/raet-cns-regression
RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/results/raet_CNS_regression/train
experiment_times=3
mode=our_Raet_CNS_test_regression
k_values=(50)
train_queries=20000
n_estimators=100
#要求准确率，搜索时的CNS大小，ipi，mpi
SIFT1M_tuples_k50=(
  "0.90 500 10"
#  "0.92 500 10"
#  "0.94 500 1"
#  "0.96 500 30"
#  "0.98 500 60"
)

GLOVE100_tuples_k50=(
  "0.90 500 10"
#  "0.92 500 10"
#  "0.94 500 10"
#  "0.96 500 20"
#  "0.98 500 50"
)

#0.96实际上参数是185
GIST1M_tuples_k50=(
  "0.90 1000 14"
#  "0.92 1000 16"
#  "0.94 1000 17"
#  "0.96 1000 18"
#  "0.98 1000 30"
)

DEEP10M_tuples_k50=(
  "0.90 750 1"
#  "0.92 750 0"
#  "0.94 750 5"
#  "0.96 750 15"
#  "0.98 750 40"
)
T2I1M_tuples_k50=(
  "0.90 1000 20"
#  "0.92 1000 35"
#  "0.94 1000 50"
#  "0.96 1000 72"
#  "0.98 1000 75"
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
        mkdir -p ${RESULTS_DIRECTORY}/${ds}/k${k}
        mkdir -p ${RESULTS_DIRECTORY}/${ds}/k${k}/times
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
        elif [ "$ds" == "T2I1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${T2I1M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${T2I1M_tuples_k50[@]}")
            fi
        fi

        for tuple in "${tuples[@]}"; do
          read target_recall efS quantile<<< "$tuple"
          echo ">>> target_recall=${target_recall}"

          spend_times=()   # 存储3次搜索时间

          for time in $(seq 1 $experiment_times); do
            echo "time=${time}"

            output=$(
              ${BUILD_DIR}/tests/test_nsg_optimized_search \
                --dataset=${ds} \
                --K=$k \
                --L=${efS} \
                --mode=${mode} \
                --query_type=train \
                --query_num=${query_num} \
                --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s10000_nestim${n_estimators}_all_feats_${target_recall}_quantile${quantile}_Noquery_Noske_Nodensty.txt \
                --out_put_path=${RESULTS_DIRECTORY}/${ds}/k${k}/times/efS${efS}_s${query_num}_tr${target_recall}_t${time}.txt \
                --target_recall=${target_recall}
            )

            echo "$output"

            # 提取搜索时间
            t=$(echo "$output" | sed -n 's/.*总搜索时间[:：]\([0-9.]*\).*/\1/p')
            spend_times+=("$t")

            echo "数据集${ds}使用达到训练集要求准确率${target_recall}的最小CNS，在测试集上进行搜索"
            echo ""
          done

          # ===== 统计 min / max / avg =====
          stats=$(printf "%s\n" "${spend_times[@]}" | awk '
            NR==1 {min=$1; max=$1; sum=0}
            {
              if ($1 < min) min=$1
              if ($1 > max) max=$1
              sum += $1
            }
            END {
              printf "min=%.5f  max=%.5f  avg=%.5f", min, max, sum/NR
            }
          ')

          echo ">>> ---------------Summary | dataset=${ds}  k=${k}  recall=${target_recall}---------------"
          echo ">>> ${stats}"
          echo ""

          python /home/extra_home/lxx23125236/ali/NSG/experiment/多次搜索取平均.py \
                ${RESULTS_DIRECTORY}/${ds}/k${k}/times/efS${efS}_s${query_num}_tr${target_recall} \
                ${RESULTS_DIRECTORY}/${ds}/k${k}/efS${efS}_s${query_num}_tr${target_recall}.txt \
                $experiment_times

          echo "多次搜索中每个查询点的平均结果，写入完成：${RESULTS_DIRECTORY}/${ds}/k${k}/efS${efS}_s${query_num}_tr${target_recall}.txt"
        done
    done
    echo ""
    echo ""
done
