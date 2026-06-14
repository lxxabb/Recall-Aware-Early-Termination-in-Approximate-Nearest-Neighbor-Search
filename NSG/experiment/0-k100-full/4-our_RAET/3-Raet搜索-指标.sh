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
dataset_params=(
  "SIFT1M 500 10000"
  "GLOVE100 500 10000"
  "GIST1M 1000 1000"
  "DEEP10M 750 10000"
)
MODEL_DIRECTORY=/root/NSG-data/predictor_models/raet
RESULTS_DIRECTORY=/root/NSG-data/results/raet
experiment_times=1
mode=our_Raet_test_metrics
k_values=(100)
train_queries=10000
n_estimators=100
#要求准确率，搜索时的CNS大小，刚好达到要求准确率的平均稳定次数  刚好达到1准确率时的平均稳定次数
SIFT1M_tuples_k100=(
  "0.95 500 62 155"
  "0.96 500 66 152"
  "0.97 500 71 146"
  "0.98 500 74 135"
  "0.99 500 69 108"
)


GLOVE100_tuples_k100=(
  "0.95 500 54 96"
  "0.96 500 53 91"
  "0.97 500 49 83"
  "0.98 500 36 63"
  "0.99 500 28 46"
)

GIST1M_tuples_k100=(
  "0.95 1000 181 394"
  "0.96 1000 198 389"
  "0.97 1000 218 380"
  "0.98 1000 237 362"
  "0.99 1000 227 306"
)

DEEP10M_tuples_k100=(
  "0.95 750 71 166"
  "0.96 750 77 163"
  "0.97 750 83 158"
  "0.98 750 89 150"
  "0.99 750 89 128"
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
          read target_recall efS stability_times stability_times_r1<<< "$tuple"
          echo ">>> target_recall=${target_recall}"
          mkdir -p ${RESULTS_DIRECTORY}/${ds}/k${k}
            ${BUILD_DIR}/tests/test_nsg_optimized_search \
              --dataset=${ds} \
              --K=$k \
              --L=${efS} \
              --mode=${mode} \
              --query_type=test \
              --query_num=${query_num} \
              --stability_times=${stability_times} \
              --stability_times_r1=${stability_times_r1} \
              --model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_nestim${n_estimators}_all_feats_Noquery_duration.txt \
              --target_recall=${target_recall} \
              --out_put_path=${RESULTS_DIRECTORY}/${ds}/k${k}/metrics_recall${target_recall}.json

        done
    done
    echo ""
    echo ""
done
#--model_path=${MODEL_DIRECTORY}/${ds}/k${k}/efS${efS_train}_s${train_queries}_nestim${n_estimators}_all_feats_Noquery_duration.txt \
#--------------- SIFT1M ---------------
#--------------- k=100 ---------------
#
#>>> target_recall=0.95
#模型预测次数:20768
#总模型预测时间(s):0.299072 seconds
#100 NN 准确率 = 0.946540
#低于要求准确率:0.950000的占比为0.397400
#平均每个数据点上的早停率:0.882200
#平均每个数据点上的成功早停率:0.485000
#平均每次预测的成功早停率:0.233532
#>>> target_recall=0.96
#模型预测次数:24494
#总模型预测时间(s):0.346564 seconds
#100 NN 准确率 = 0.962377
#低于要求准确率:0.960000的占比为0.338000
#平均每个数据点上的早停率:0.801000
#平均每个数据点上的成功早停率:0.463400
#平均每次预测的成功早停率:0.189189
#>>> target_recall=0.97
#模型预测次数:27096
#总模型预测时间(s):0.387026 seconds
#100 NN 准确率 = 0.977407
#低于要求准确率:0.970000的占比为0.255300
#平均每个数据点上的早停率:0.674300
#平均每个数据点上的成功早停率:0.419500
#平均每次预测的成功早停率:0.154820
#>>> target_recall=0.98
#模型预测次数:29349
#总模型预测时间(s):0.428551 seconds
#100 NN 准确率 = 0.988125
#低于要求准确率:0.980000的占比为0.175100
#平均每个数据点上的早停率:0.515500
#平均每个数据点上的成功早停率:0.342900
#平均每次预测的成功早停率:0.116835
#>>> target_recall=0.99
#模型预测次数:34851
#总模型预测时间(s):0.457049 seconds
#100 NN 准确率 = 0.993149
#低于要求准确率:0.990000的占比为0.152400
#平均每个数据点上的早停率:0.330600
#平均每个数据点上的成功早停率:0.186200
#平均每次预测的成功早停率:0.053427
#
#
#
#--------------- GLOVE100 ---------------
#
#
#--------------- k=100 ---------------
#
#>>> target_recall=0.95
#tcmalloc: large alloc 1358675968 bytes == 0x556d24542000 @
#模型预测次数:12487
#总模型预测时间(s):0.140702 seconds
#100 NN 准确率 = 0.913886
#低于要求准确率:0.950000的占比为0.698300
#平均每个数据点上的早停率:0.960900
#平均每个数据点上的成功早停率:0.277200
#平均每次预测的成功早停率:0.221991
#>>> target_recall=0.96
#tcmalloc: large alloc 1358675968 bytes == 0x5583d8624000 @
#模型预测次数:14235
#总模型预测时间(s):0.169668 seconds
#100 NN 准确率 = 0.913084
#低于要求准确率:0.960000的占比为0.788700
#平均每个数据点上的早停率:0.935400
#平均每个数据点上的成功早停率:0.165500
#平均每次预测的成功早停率:0.116263
#>>> target_recall=0.97
#tcmalloc: large alloc 1358675968 bytes == 0x558382dba000 @
#模型预测次数:17008
#总模型预测时间(s):0.184914 seconds
#100 NN 准确率 = 0.902900
#低于要求准确率:0.970000的占比为0.882400
#平均每个数据点上的早停率:0.905000
#平均每个数据点上的成功早停率:0.047100
#平均每次预测的成功早停率:0.027693
#>>> target_recall=0.98
#tcmalloc: large alloc 1358675968 bytes == 0x55d0accde000 @
#模型预测次数:34410
#总模型预测时间(s):0.313434 seconds
#100 NN 准确率 = 0.865345
#低于要求准确率:0.980000的占比为0.884300
#平均每个数据点上的早停率:0.858200
#平均每个数据点上的成功早停率:0.005100
#平均每次预测的成功早停率:0.001482
#>>> target_recall=0.99
#tcmalloc: large alloc 1358675968 bytes == 0x557999d6c000 @
#模型预测次数:84232
#总模型预测时间(s):0.657529 seconds
#100 NN 准确率 = 0.881465
#低于要求准确率:0.990000的占比为0.800500
#平均每个数据点上的早停率:0.766300
#平均每个数据点上的成功早停率:0.012200
#平均每次预测的成功早停率:0.001448
#
#
#
#--------------- GIST1M ---------------
#
#
#--------------- k=100 ---------------
#
#>>> target_recall=0.95
#tcmalloc: large alloc 3840000000 bytes == 0x55b1b8f38000 @
#tcmalloc: large alloc 4172005376 bytes == 0x55b2a5174000 @
#tcmalloc: large alloc 3840000000 bytes == 0x55b39e50c000 @
#模型预测次数:2562
#总模型预测时间(s):0.068875 seconds
#100 NN 准确率 = 0.992008
#低于要求准确率:0.950000的占比为0.021000
#平均每个数据点上的早停率:0.389000
#平均每个数据点上的成功早停率:0.375000
#平均每次预测的成功早停率:0.146370
#>>> target_recall=0.96
#tcmalloc: large alloc 3840000000 bytes == 0x5620f5f2c000 @
#tcmalloc: large alloc 4172005376 bytes == 0x5621e2168000 @
#tcmalloc: large alloc 3840000000 bytes == 0x5622db500000 @
#模型预测次数:2287
#总模型预测时间(s):0.063072 seconds
#100 NN 准确率 = 0.994138
#低于要求准确率:0.960000的占比为0.017000
#平均每个数据点上的早停率:0.306000
#平均每个数据点上的成功早停率:0.303000
#平均每次预测的成功早停率:0.132488
#>>> target_recall=0.97
#tcmalloc: large alloc 3840000000 bytes == 0x557a9f90c000 @
#tcmalloc: large alloc 4172005376 bytes == 0x557b8bb48000 @
#tcmalloc: large alloc 3840000000 bytes == 0x557c84ee0000 @
#模型预测次数:2078
#总模型预测时间(s):0.063375 seconds
#100 NN 准确率 = 0.994968
#低于要求准确率:0.970000的占比为0.026000
#平均每个数据点上的早停率:0.242000
#平均每个数据点上的成功早停率:0.239000
#平均每次预测的成功早停率:0.115014
#>>> target_recall=0.98
#tcmalloc: large alloc 3840000000 bytes == 0x55cc7b950000 @
#tcmalloc: large alloc 4172005376 bytes == 0x55cd67b8c000 @
#tcmalloc: large alloc 3840000000 bytes == 0x55ce60f24000 @
#模型预测次数:1887
#总模型预测时间(s):0.056440 seconds
#100 NN 准确率 = 0.995639
#低于要求准确率:0.980000的占比为0.037000
#平均每个数据点上的早停率:0.170000
#平均每个数据点上的成功早停率:0.168000
#平均每次预测的成功早停率:0.089030
#>>> target_recall=0.99
#tcmalloc: large alloc 3840000000 bytes == 0x564a3d142000 @
#tcmalloc: large alloc 4172005376 bytes == 0x564b2937e000 @
#tcmalloc: large alloc 3840000000 bytes == 0x564c22716000 @
#模型预测次数:1914
#总模型预测时间(s):0.056778 seconds
#100 NN 准确率 = 0.995809
#低于要求准确率:0.990000的占比为0.084000
#平均每个数据点上的早停率:0.108000
#平均每个数据点上的成功早停率:0.106000
#平均每次预测的成功早停率:0.055381
#
#
#
#--------------- DEEP10M ---------------
#
#
#--------------- k=100 ---------------
#
#>>> target_recall=0.95
#tcmalloc: large alloc 3840000000 bytes == 0x5584cf898000 @
#tcmalloc: large alloc 6880002048 bytes == 0x558632a1a000 @
#tcmalloc: large alloc 3840000000 bytes == 0x5587cd3f8000 @
#模型预测次数:20119
#总模型预测时间(s):0.397322 seconds
#100 NN 准确率 = 0.968741
#低于要求准确率:0.950000的占比为0.198400
#平均每个数据点上的早停率:0.833100
#平均每个数据点上的成功早停率:0.640400
#平均每次预测的成功早停率:0.318306
#>>> target_recall=0.96
#tcmalloc: large alloc 3840000000 bytes == 0x55a368126000 @
#tcmalloc: large alloc 6880002048 bytes == 0x55a4cb2a8000 @
#tcmalloc: large alloc 3840000000 bytes == 0x55a665c86000 @
#模型预测次数:21150
#总模型预测时间(s):0.408252 seconds
#100 NN 准确率 = 0.978483
#低于要求准确率:0.960000的占比为0.160800
#平均每个数据点上的早停率:0.776700
#平均每个数据点上的成功早停率:0.624000
#平均每次预测的成功早停率:0.295035
#>>> target_recall=0.97
#tcmalloc: large alloc 3840000000 bytes == 0x556fdf2c2000 @
#tcmalloc: large alloc 6880002048 bytes == 0x557142444000 @
#tcmalloc: large alloc 3840000000 bytes == 0x5572dce22000 @
#模型预测次数:21944
#总模型预测时间(s):0.435007 seconds
#100 NN 准确率 = 0.986314
#低于要求准确率:0.970000的占比为0.131000
#平均每个数据点上的早停率:0.699000
#平均每个数据点上的成功早停率:0.582500
#平均每次预测的成功早停率:0.265448
#>>> target_recall=0.98
#tcmalloc: large alloc 3840000000 bytes == 0x562fc6ab8000 @
#tcmalloc: large alloc 6880002048 bytes == 0x563129c3a000 @
#tcmalloc: large alloc 3840000000 bytes == 0x5632c4618000 @
#模型预测次数:23448
#总模型预测时间(s):0.451648 seconds
#100 NN 准确率 = 0.993110
#低于要求准确率:0.980000的占比为0.085500
#平均每个数据点上的早停率:0.564100
#平均每个数据点上的成功早停率:0.500900
#平均每次预测的成功早停率:0.213622
#>>> target_recall=0.99
#tcmalloc: large alloc 3840000000 bytes == 0x564875f1c000 @
#tcmalloc: large alloc 6880002048 bytes == 0x5649d909e000 @
#tcmalloc: large alloc 3840000000 bytes == 0x564b73a7c000 @
#模型预测次数:24860
#总模型预测时间(s):0.497175 seconds
#100 NN 准确率 = 0.995946
#低于要求准确率:0.990000的占比为0.081800
#平均每个数据点上的早停率:0.391500
#平均每个数据点上的成功早停率:0.349100
#平均每次预测的成功早停率:0.140426
