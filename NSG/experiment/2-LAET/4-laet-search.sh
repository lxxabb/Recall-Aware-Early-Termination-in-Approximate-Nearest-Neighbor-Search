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
  "GLOVE100 500 10000"
  "GIST1M 1000 1000"
  "DEEP10M 750 10000"
)
MODEL_DIRECTORY=/root/NSG-data/predictor_models/laet
experiment_times=1
mode=Laet_test
k_values=(100)
train_queries=10000
#F参数，是80%的的数据点达到最高准确率的距离计算次数
SIFT1M_tuples_k100=(
  "0.95 100 2954 1"
  "0.96 100 2954 1"
  "0.97 106 2954 1"
  "0.98 128 2954 1"
  "0.99 174 2954 1.5"
)


GLOVE100_tuples_k100=(
  "0.95 100 1028 1"
  "0.96 100 1028 1"
  "0.97 100 1028 1"
  "0.98 100 1028 1"
  "0.99 210 1028 3"
)

GIST1M_tuples_k100=(
  "0.95 220 4495 1.5"
  "0.96 255 4495 1.5"
  "0.97 306 4495 5"
  "0.98 394 4495 3.5"
  "0.99 591 4495 6"
)

DEEP10M_tuples_k100=(
  "0.95 123 3470 1"
  "0.96 148 3470 1"
  "0.97 188 3470 1"
  "0.98 257 3470 2.5"
  "0.99 432 3470 5"
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
          read target_recall efS laet_F laet_multiplier <<< "$tuple"
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
                --laet_F=${laet_F} \
                --laet_multiplier=${laet_multiplier} \
                --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_f${laet_F}.txt \
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
