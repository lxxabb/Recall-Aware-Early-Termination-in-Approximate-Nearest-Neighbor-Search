#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e

RESULT_DATA_DIRECTORY=/home/extra_home/lxx23125236/ali/NSG-data/results/raet

#ds M efC efSFULL sample
dataset_params=(
#  "SIFT1M 32 500 500 10000"
#  "GLOVE100 16 500 500 10000"
#  "GIST1M 32 500 1000 10000"
#  "DEEP10M 32 500 750 10000"
  "T2I1M 80 1000 1000 10000"
)


k_values=(50)


SIFT1M_tuples_k50=(
  "0.90 500"
  "0.92 500"
  "0.94 500"
  "0.96 500"
  "0.98 500"
)


GLOVE100_tuples_k50=(
  "0.90 500"
  "0.92 500"
  "0.94 500"
  "0.96 500"
  "0.98 500"
)

#gist数据集700 的准确率为0.9887
GIST1M_tuples_k50=(
  "0.90 1000"
  "0.92 1000"
  "0.94 1000"
  "0.96 1000"
  "0.98 1000"
)
#400 准确率为0.9951
DEEP10M_tuples_k50=(
  "0.90 750"
  "0.92 750"
  "0.94 750"
  "0.96 750"
  "0.98 750"
)
T2I1M_tuples_k50=(
  "0.90 1000"
  "0.92 1000"
  "0.94 1000"
  "0.96 1000"
  "0.98 1000"
)
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFULL sample <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

#    mkdir -p ${RESULTS_DIRECTORY}/REM/${ds}

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
        elif [ "$ds" == "DEEP10M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP10M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${DEEP10M_tuples_k50[@]}")
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
        elif [ "$ds" == "T2I1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${T2I1M_tuples_k100[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${T2I1M_tuples_k50[@]}")
            fi
        fi
        for tuple in "${tuples[@]}"; do
          read target_recall efS <<< "$tuple"
          echo ">>> target_recall=${target_recall}, efS=${efS}"
          if true; then
            python /home/extra_home/lxx23125236/ali/NSG/experiment/0-k50-full/4-our_RAET/1-1-刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall $target_recall
            python /home/extra_home/lxx23125236/ali/NSG/experiment/0-k50-full/4-our_RAET/1-1-刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall 1.0
          fi
        done
    done
    echo ""
    echo ""
done
