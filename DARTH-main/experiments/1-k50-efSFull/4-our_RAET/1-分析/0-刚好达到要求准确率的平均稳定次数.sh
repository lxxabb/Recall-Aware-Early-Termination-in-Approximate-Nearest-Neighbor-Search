#!/usr/bin/env bash
#脚本里只要有一个命令失败（返回值非 0），整个脚本就立刻退出
set -e

RESULTS_DIRECTORY=/home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/


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

    mkdir -p ${RESULTS_DIRECTORY}/REM/${ds}

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
            python /home/extra_home/lxx23125236/ali/DARTH-main/experiments/1-k50-efSFull/4-our_RAET/1-分析/刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULTS_DIRECTORY}${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall $target_recall
            python /home/extra_home/lxx23125236/ali/DARTH-main/experiments/1-k50-efSFull/4-our_RAET/1-分析/刚好达到要求准确率的平均稳定次数.py \
                  --file ${RESULTS_DIRECTORY}${ds}/k${k}/stablity/efS${efS}_qs${sample}_tr${target_recall}.txt \
                  --target_recall 1.0
          fi
        done
    done
    echo ""
    echo ""
done
#--------------- SIFT1M ---------------
#
#
#--------------- k=50 ---------------

#>>> target_recall=0.90, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.90.txt
#target_recall : 0.9
#mean          : 27.508
#median        : 23.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.90.txt
#target_recall : 1.0
#mean          : 96.629
#median        : 76.000
#>>> target_recall=0.92, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.92.txt
#target_recall : 0.92
#mean          : 29.057
#median        : 24.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.92.txt
#target_recall : 1.0
#mean          : 93.419
#median        : 73.000
#>>> target_recall=0.94, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.94.txt
#target_recall : 0.94
#mean          : 35.513
#median        : 29.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.94.txt
#target_recall : 1.0
#mean          : 93.419
#median        : 73.000
#>>> target_recall=0.96, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 38.555
#median        : 31.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 87.293
#median        : 67.000
#>>> target_recall=0.98, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 38.688
#median        : 27.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/SIFT1M/k50/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 72.249
#median        : 52.000
#
#
#
#--------------- GLOVE100 ---------------
#
#
#--------------- k=50 ---------------
#
#>>> target_recall=0.90, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.90.txt
#target_recall : 0.9
#mean          : 23.478
#median        : 17.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.90.txt
#target_recall : 1.0
#mean          : 67.690
#median        : 42.000
#>>> target_recall=0.92, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.92.txt
#target_recall : 0.92
#mean          : 23.003
#median        : 15.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.92.txt
#target_recall : 1.0
#mean          : 63.473
#median        : 38.000
#>>> target_recall=0.94, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.94.txt
#target_recall : 0.94
#mean          : 26.789
#median        : 18.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.94.txt
#target_recall : 1.0
#mean          : 63.473
#median        : 38.000
#>>> target_recall=0.96, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 27.161
#median        : 16.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 57.717
#median        : 33.000
#>>> target_recall=0.98, efS=500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 27.295
#median        : 11.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GLOVE100/k50/stablity/efS500_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 48.000
#median        : 24.000
#
#
#
#--------------- GIST1M ---------------
#
#
#--------------- k=50 ---------------
#
#>>> target_recall=0.90, efS=1000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.90.txt
#target_recall : 0.9
#mean          : 109.527
#median        : 83.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.90.txt
#target_recall : 1.0
#mean          : 323.112
#median        : 260.000
#>>> target_recall=0.92, efS=1000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.92.txt
#target_recall : 0.92
#mean          : 123.537
#median        : 91.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.92.txt
#target_recall : 1.0
#mean          : 317.536
#median        : 254.000
#>>> target_recall=0.94, efS=1000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.94.txt
#target_recall : 0.94
#mean          : 150.093
#median        : 111.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.94.txt
#target_recall : 1.0
#mean          : 317.536
#median        : 254.000
#>>> target_recall=0.96, efS=1000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 173.274
#median        : 125.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 306.245
#median        : 244.000
#>>> target_recall=0.98, efS=1000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 188.783
#median        : 130.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/GIST1M/k50/stablity/efS1000_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 270.923
#median        : 207.000

#DEEP
#>>> target_recall=0.90, efS=750
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.90.txt
#target_recall : 0.9
#mean          : 32.119
#median        : 20.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.90.txt
#target_recall : 1.0
#mean          : 123.417
#median        : 80.000
#>>> target_recall=0.92, efS=750
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.92.txt
#target_recall : 0.92
#mean          : 36.206
#median        : 22.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.92.txt
#target_recall : 1.0
#mean          : 120.806
#median        : 77.000
#>>> target_recall=0.94, efS=750
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.94.txt
#target_recall : 0.94
#mean          : 45.216
#median        : 27.500
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.94.txt
#target_recall : 1.0
#mean          : 120.806
#median        : 77.000
#>>> target_recall=0.96, efS=750
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.96.txt
#target_recall : 0.96
#mean          : 52.594
#median        : 31.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.96.txt
#target_recall : 1.0
#mean          : 115.602
#median        : 72.000
#>>> target_recall=0.98, efS=750
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.98.txt
#target_recall : 0.98
#mean          : 58.752
#median        : 30.000
#===== Stability statistics =====
#file          : /home/extra_home/lxx23125236/ali/DARTH-main-data/result/raet/DEEP10M/k50/stablity/efS750_qs10000_tr0.98.txt
#target_recall : 1.0
#mean          : 101.139
#median        : 60.000
#
