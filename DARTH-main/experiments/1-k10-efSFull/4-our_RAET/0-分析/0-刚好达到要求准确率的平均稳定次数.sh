#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e

RESULTS_DIRECTORY=/root/DARTH-main-data/result/raet/


#ds M efC efSFULL sample
dataset_params=(
  "SIFT1M 32 500 500 10000"
  "GLOVE100 16 500 500 10000"
  "GIST1M 32 500 1000 10000"
  "DEEP10M 32 500 750 10000"
)


k_values=(100)


SIFT1M_tuples_k100=(
  "0.95 500"
  "0.96 500"
  "0.97 500"
  "0.98 500"
  "0.99 500"
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
            python /root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/0-分析/刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULTS_DIRECTORY}${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall $target_recall
            python /root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/0-分析/刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULTS_DIRECTORY}${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall 1.0
          fi
        done
    done
    echo ""
    echo ""
done
#--------------- SIFT1M ---------------
#--------------- k=100 ---------------
#>>> target_recall=0.95, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.95.txt
#target_recall : 0.95
#mean          : 60.351
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.95.txt
#target_recall : 1.0
#mean          : 157.108
#>>> target_recall=0.96, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 64.267
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 153.323
#>>> target_recall=0.97, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.97.txt
#target_recall : 0.97
#mean          : 68.318
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.97.txt
#target_recall : 1.0
#mean          : 147.327
#>>> target_recall=0.98, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 71.545
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 136.539
#>>> target_recall=0.99, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.99.txt
#target_recall : 0.99
#mean          : 67.158
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/SIFT1M/k100/stablity/efS500_qs10000_tr0.99.txt
#target_recall : 1.0
#mean          : 110.321
#
#
#
#--------------- GLOVE100 ---------------
#--------------- k=100 ---------------
#>>> target_recall=0.95, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.95.txt
#target_recall : 0.95
#mean          : 56.701
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.95.txt
#target_recall : 1.0
#mean          : 106.125
#>>> target_recall=0.96, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 53.724
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 98.683
#>>> target_recall=0.97, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.97.txt
#target_recall : 0.97
#mean          : 42.319
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.97.txt
#target_recall : 1.0
#mean          : 81.237
#>>> target_recall=0.98, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 37.065
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 68.995
#>>> target_recall=0.99, efS=500
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.99.txt
#target_recall : 0.99
#mean          : 33.402
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GLOVE100/k100/stablity/efS500_qs10000_tr0.99.txt
#target_recall : 1.0
#mean          : 55.002
#
#--------------- GIST1M ---------------
#--------------- k=100 ---------------
#>>> target_recall=0.95, efS=1000
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.95.txt
#target_recall : 0.95
#mean          : 221.581
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.95.txt
#target_recall : 1.0
#mean          : 421.572
#>>> target_recall=0.96, efS=1000
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 241.867
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 415.960
#>>> target_recall=0.97, efS=1000
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.97.txt
#target_recall : 0.97
#mean          : 265.060
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.97.txt
#target_recall : 1.0
#mean          : 406.483
#>>> target_recall=0.98, efS=1000
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 281.317
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 387.307
#>>> target_recall=0.99, efS=1000
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.99.txt
#target_recall : 0.99
#mean          : 267.789
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/GIST1M/k100/stablity/efS1000_qs10000_tr0.99.txt
#target_recall : 1.0
#mean          : 332.496
#
#--------------- DEEP10M ---------------
#--------------- k=100 ---------------
#>>> target_recall=0.95, efS=750
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.95.txt
#target_recall : 0.95
#mean          : 82.936
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.95.txt
#target_recall : 1.0
#mean          : 202.093
#>>> target_recall=0.96, efS=750
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 89.980
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 198.597
#>>> target_recall=0.97, efS=750
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.97.txt
#target_recall : 0.97
#mean          : 98.495
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.97.txt
#target_recall : 1.0
#mean          : 193.073
#>>> target_recall=0.98, efS=750
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 105.457
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 182.568
#>>> target_recall=0.99, efS=750
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.99.txt
#target_recall : 0.99
#mean          : 105.843
#===== Stability statistics =====
#file          : /root/DARTH-main-data/result/raet/DEEP10M/k100/stablity/efS750_qs10000_tr0.99.txt
#target_recall : 1.0
#mean          : 155.413
#
#
#
