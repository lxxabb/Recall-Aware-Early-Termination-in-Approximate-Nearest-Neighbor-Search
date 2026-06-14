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
#  "SIFT1M 500 10000"
#  "GLOVE100 500 10000"
#  "GIST1M 1000 1000"
  "DEEP10M 750 10000"
)
MODEL_DIRECTORY=/root/NSG-data/predictor_models/darth
experiment_times=1
mode=Darth_test_time
k_values=(100)
train_queries=10000
n_estimators=100
#要求准确率，搜索时的CNS大小，ipi，mpi
SIFT1M_tuples_k100=(
  "0.95 100 2200 265"
  "0.96 100 2726 140"
  "0.97 106 3400 306"
  "0.98 128 4100 336"
  "0.99 174 5000 381"
)

GLOVE100_tuples_k100=(
  "0.95 100 628 126"
  "0.96 100 642 128"
  "0.97 100 660 132"
  "0.98 100 690 138"
  "0.99 210 733 147"
)

GIST1M_tuples_k100=(
  "0.95 220 9000 777"
  "0.96 255 10000 840"
  "0.97 306 12000 918"
  "0.98 394 15000 1013"
  "0.99 591 21000 1133"
)

DEEP10M_tuples_k100=(
#  "0.95 123 3700 413"
#  "0.96 148 4200 442"
#  "0.97 188 5500 477"
#  "0.98 257 8500 518"
  "0.99 432 13000 573"
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
                --ipi=${ipi} \
                --mpi=${mpi} \
                --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_nestim${n_estimators}_all_feats.txt \
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
#SIFT1M_tuples_k100=(
#  "0.95 100 1324 265"
#  "0.96 100 1413 283"
#  "0.97 106 1528 306"
#  "0.98 128 1681 336"
#  "0.99 174 1904 381"
#)
#
#
#GLOVE100_tuples_k100=(
#  "0.95 100 628 126"
#  "0.96 100 642 128"
#  "0.97 100 660 132"
#  "0.98 100 690 138"
#  "0.99 210 733 147"
#)
#
#GIST1M_tuples_k100=(
#  "0.95 220 3886 777"
#  "0.96 255 4200 840"
#  "0.97 306 4590 918"
#  "0.98 394 5067 1013"
#  "0.99 591 5663 1133"
#)
#
#DEEP10M_tuples_k100=(
#  "0.95 123 2066 413"
#  "0.96 148 2211 442"
#  "0.97 188 2384 477"
#  "0.98 257 2590 518"
#  "0.99 432 2864 573"
#)
