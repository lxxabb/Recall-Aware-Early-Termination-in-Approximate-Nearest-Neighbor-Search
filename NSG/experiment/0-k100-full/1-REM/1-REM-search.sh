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
#  -DCMAKE_BUILD_TYPE=Release \
#
## -------- 2️⃣ 编译 --------
#ninja -C ${BUILD_DIR} test_nsg_optimized_search
#
#
#echo ""
#echo "============= Build Done (Release) ============="
#echo ""
#构建knn图索引命令
#sift  ./test_nndescent /home/extra_home/lxx23125236/ali/ann-data/SIFT1M/sift_base.fvecs /home/extra_home/lxx23125236/ali/NSG-data/sift_200nn.graph 200 200 10 10 100
#glove ./test_nndescent /home/extra_home/lxx23125236/ali/ann-data/GLOVE100/base.1183514.fvecs /home/extra_home/lxx23125236/ali/NSG-data/glove-100_400nn.knng 400 420 12 15 200
#gist ./test_nndescent /home/extra_home/lxx23125236/ali/ann-data/GIST1M/gist_base.fvecs /home/extra_home/lxx23125236/ali/NSG-data/gist_400nn.graph 400 400 12 15 100
#deep ./test_nndescent /home/extra_home/lxx23125236/ali/ann-data/DEEP10M/deep10M.fvecs /home/extra_home/lxx23125236/ali/NSG-data/deep_400nn.graph 400 400 12 15 100
#T2I ./test_nndescent /home/extra_home/lxx23125236/ann-data/T2I1M/base.1B.fbin.fvecs /home/extra_home/lxx23125236/ali/NSG-data/t2i_400nn.graph 400 420 12 15 200
#构建nsg图索引命令
#sift.fvecs sift_200nn.graph 40 50 500 sift_40_50_500.nsg
#Degree Statistics: Max = 50, Min = 1, Avg = 29
#indexing time: 167.17

#gist_base.fvecs gist_400nn.graph 60 70 500 gist_60_70_500.nsg
#Degree Statistics: Max = 81, Min = 1, Avg = 21
#indexing time: 1124.69

#base.1183514.fvecs glove-100_400nn.graph 60 70 500 glove_60_70_500.nsg
#Degree Statistics: Max = 185, Min = 1, Avg = 12
#indexing time: 236.194

#deep10M.fvecs deep_400nn.graph 60 70 500 deep_60_70_500.nsg
#Degree Statistics: Max = 74, Min = 1, Avg = 40
#indexing time: 4368.86

#base.1B.fbin.fvecs t2i_400nn.graph 60 70 500 t2i_60_70_500.nsg
#Degree Statistics: Max = 70, Min = 1, Avg = 31
#indexing time: 325.024

#sift 500   gist 500
#数据集和每个数据集中  测试集中的查询点数量
dataset_params=(
#  "SIFT1M 10000"
#  "GLOVE100 10000"
#  "GIST1M 1000"
#  "DEEP10M 10000"
  "T2I1M 10000"
)
RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/results/REM
experiment_times=3
mode=REM_test
k_values=(100)


SIFT1M_tuples_k100=(
  "0.95 500"
#  "0.96 500"
#  "0.97 500"
#  "0.98 500"
#  "0.99 500"
)


GLOVE100_tuples_k100=(
  "0.95 500"
#  "0.96 500"
#  "0.97 500"
#  "0.98 500"
#  "0.99 500"
)


GIST1M_tuples_k100=(
  "0.95 1000"
#  "0.96 1000"
#  "0.97 1000"
#  "0.98 1000"
#  "0.99 1000"
)

DEEP10M_tuples_k100=(
  "0.95 750"
#  "0.96 750"
#  "0.97 750"
#  "0.98 750"
#  "0.99 750"
)
T2I1M_tuples_k100=(
  "0.95 1000"
#  "0.96 1000"
#  "0.97 1000"
#  "0.98 1000"
#  "0.99 1000"
)

for dataset_param in "${dataset_params[@]}"
do
    read ds query_num<<< "$dataset_param"

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
        elif [ "$ds" == "T2I1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${T2I1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${T2I1M_tuples_k10[@]}")
            fi
        fi

        for tuple in "${tuples[@]}"; do
          read target_recall efS <<< "$tuple"
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
                --out_put_path=${RESULTS_DIRECTORY}/${ds}/k${k}/efS${efS}_s${query_num}_tr${target_recall}.txt \
                --query_num=${query_num}
            )

            echo "$output"

            # 提取搜索时间
            t=$(echo "$output" | awk -F'[：:]' '/总搜索时间/ {print $2}')
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
