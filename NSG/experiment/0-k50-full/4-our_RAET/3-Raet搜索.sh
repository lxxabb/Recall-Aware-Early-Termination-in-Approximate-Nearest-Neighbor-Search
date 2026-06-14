#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e
export LD_LIBRARY_PATH=/home/extra_home/lxx23125236/ali/LightGBM/lib:$LD_LIBRARY_PATH
# -------- 1️⃣ 构建配置 --------
BUILD_DIR=/home/extra_home/lxx23125236/ali/NSG/cmake-build-lxx-release

#/home/lxx23125236/.cache/JetBrains/RemoteDev/dist/0a55911a5a2e1_CLion-2024.1.1/bin/cmake/linux/x64/bin/cmake \
#  -S /home/extra_home/lxx23125236/ali/NSG \
#  -B ${BUILD_DIR} \
#  -G Ninja \
#  -DCMAKE_BUILD_TYPE=Release
#
## -------- 2️⃣ 编译 --------
#ninja -C ${BUILD_DIR} test_nsg_optimized_search
#
#
#echo ""
#echo "============= Build Done (Release) ============="
#echo ""
#数据集和每个数据集中  构建训练数据集时采用的efS 测试集中的查询点数量
dataset_params=(
  "SIFT1M 500 10000"
  "GLOVE100 500 10000"
  "GIST1M 1000 1000"
  "DEEP10M 750 10000"
#  "T2I1M 1000 10000"
)
MODEL_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/predictor_models/raet
RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/results/raet
experiment_times=3
mode=our_Raet_test
k_values=(50)
train_queries=10000
n_estimators=100
#要求准确率，搜索时的CNS大小，ipi，mpi
SIFT1M_tuples_k50=(
  "0.90 500 30 100"
#  "0.92 500 33 98"
#  "0.94 500 36 95"
#  "0.96 500 40 89"
#  "0.98 500 40 74"
)

GLOVE100_tuples_k50=(
  "0.90 500 24 62"
#  "0.92 500 25 59"
#  "0.94 500 25 56"
#  "0.96 500 25 50"
#  "0.98 500 23 41"
)

GIST1M_tuples_k50=(
  "0.90 1000 92 287"
#  "0.92 1000 104 284"
#  "0.94 1000 120 279"
#  "0.96 1000 140 268"
#  "0.98 1000 151 234"
)

DEEP10M_tuples_k50=(
  "0.90 750 29 104"
#  "0.92 750 34 103"
#  "0.94 750 39 100"
#  "0.96 750 46 96"
#  "0.98 750 51 85"
)
#T2I1M_tuples_k50=(
#  "0.90 1000 83 242"
#  "0.92 1000 93 239"
#  "0.94 1000 106 234"
#  "0.96 1000 122 225"
#  "0.98 1000 132 202"
#)
T2I1M_tuples_k50=(
  "0.90 1000 93 242"
  "0.92 1000 110 254"
  "0.94 1000 106 234"
  "0.96 1000 122 225"
  "0.98 1000 132 202"
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
          read target_recall efS stability_times stability_times_r1<<< "$tuple"
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
                --query_type=test \
                --query_num=${query_num} \
                --stability_times=${stability_times} \
                --stability_times_r1=${stability_times_r1} \
                --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_nestim${n_estimators}_all_feats_r_ge_0.90.txt \
                --out_put_path=${RESULTS_DIRECTORY}/${ds}/k${k}/efS${efS}_s${query_num}_tr${target_recall}.txt \
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
        done
    done
    echo ""
    echo ""
done
