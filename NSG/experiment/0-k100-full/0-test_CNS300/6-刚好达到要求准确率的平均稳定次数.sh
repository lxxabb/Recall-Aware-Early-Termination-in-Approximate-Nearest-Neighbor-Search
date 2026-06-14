#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e

RESULT_DATA_DIRECTORY=/root/NSG-data/results/raet

#ds M efC efSFULL sample
dataset_params=(
  "SIFT1M 32 500 300 10000"
#  "GLOVE100 16 500 500 10000"
#  "GIST1M 32 500 1000 10000"
#  "DEEP10M 32 500 750 10000"
)


k_values=(100)


SIFT1M_tuples_k100=(
  "0.95 300"
  "0.96 300"
  "0.97 300"
  "0.98 300"
  "0.99 300"
)


GLOVE100_tuples_k100=(
  "0.95 500"
  "0.96 500"
  "0.97 500"
  "0.98 500"
  "0.99 500"
)

#gist数据集700 的准确率为0.9887
GIST1M_tuples_k100=(
  "0.95 1000"
  "0.96 1000"
  "0.97 1000"
  "0.98 1000"
  "0.99 1000"
)
#400 准确率为0.9951
DEEP10M_tuples_k100=(
  "0.95 750"
  "0.96 750"
  "0.97 750"
  "0.98 750"
  "0.99 750"
)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFULL sample <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir -p ${RESULTS_DIRECTORY}/REM/${ds}

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
          read target_recall efS <<< "$tuple"
          echo ">>> target_recall=${target_recall}, efS=${efS}"
          if true; then
            python /root/NSG/experiment/0-k100-full/4-our_RAET/1-1-刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall $target_recall
            python /root/NSG/experiment/0-k100-full/4-our_RAET/1-1-刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULT_DATA_DIRECTORY}/${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall 1.0
          fi
        done
    done
    echo ""
    echo ""
done
--------------- SIFT1M ---------------


--------------- k=100 ---------------

>>> target_recall=0.95, efS=300
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.95.txt
target_recall : 0.95
mean          : 61.737
median        : 53.000
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.95.txt
target_recall : 1.0
mean          : 129.629
median        : 119.000
>>> target_recall=0.96, efS=300
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.96.txt
target_recall : 0.96
mean          : 65.447
median        : 55.000
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.96.txt
target_recall : 1.0
mean          : 126.151
median        : 115.000
>>> target_recall=0.97, efS=300
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.97.txt
target_recall : 0.97
mean          : 68.919
median        : 58.000
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.97.txt
target_recall : 1.0
mean          : 120.762
median        : 110.000
>>> target_recall=0.98, efS=300
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.98.txt
target_recall : 0.98
mean          : 69.733
median        : 58.000
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.98.txt
target_recall : 1.0
mean          : 110.460
median        : 99.000
>>> target_recall=0.99, efS=300
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.99.txt
target_recall : 0.99
mean          : 60.030
median        : 48.000
===== Stability statistics =====
file          : /root/NSG-data/results/raet/SIFT1M/k100/stablity/efS300_qs10000_tr0.99.txt
target_recall : 1.0
mean          : 85.817
median        : 74.000
