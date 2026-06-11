#!/bin/bash

LOG_DIR="./logs_naswot_1000"
DATA_PATH_CIFAR="./cifar.python"
DATA_PATH_IMAGENET="./cifar.python/ImageNet16-120"
export CUDA_VISIBLE_DEVICES=0

METRICS=(
#    "singular"
#    "gradnorm"
#    "synflow"
#    "zen"
#    "zico"
#    "entropic"
#    "naswot"
#    "grasp"
#    "snip"
#    "te_nas"
)
mkdir -p $LOG_DIR

echo " $(date)"
echo "======================================"


count=0
total_runs=$(( ${#METRICS[@]} * 3 * 2 ))

for metric in "${METRICS[@]}"; do

    echo " $metric, seed:1"

    # cifar10
    echo "  -> cifar10"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar10 \
        --rand_seed 1
    ((count++))
    echo " $count/$total_runs"
    echo "--------------------------------------"

    # cifar100
    echo "  -> cifar100"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar100 \
        --rand_seed 1
    ((count++))
    echo " $count/$total_runs"
    echo "--------------------------------------"

    # ImageNet16-120
    echo "  -> ImageNet16-120"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_IMAGENET \
        --dataset ImageNet16-120 \
        --rand_seed 1
    ((count++))
    echo "$count/$total_runs"
    echo "--------------------------------------"

    echo " $metric, seed: 2"

    # cifar10
    echo "  -> cifar10"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar10 \
        --rand_seed 2
    ((count++))
    echo " $count/$total_runs"
    echo "--------------------------------------"

    # cifar100
    echo "  -> cifar100"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar100 \
        --rand_seed 2
    ((count++))
    echo " $count/$total_runs"
    echo "--------------------------------------"

    # ImageNet16-120
    echo "  -> ImageNet16-120"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_IMAGENET \
        --dataset ImageNet16-120 \
        --rand_seed 2
    ((count++))
    echo " $count/$total_runs"
    echo "======================================"


    echo " $metric, seed: 3"

    # cifar10
    echo "  -> cifar10"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar10 \
        --rand_seed 3
    ((count++))
    echo "$count/$total_runs"
    echo "--------------------------------------"

    # cifar100
    echo "  -> cifar100"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_CIFAR \
        --dataset cifar100 \
        --rand_seed 3
    ((count++))
    echo "$count/$total_runs"
    echo "--------------------------------------"

    # ImageNet16-120
    echo "  -> ImageNet16-120"
    python eval.py \
        --gpu 0 \
        --zero_cost_score $metric \
        --log_dir $LOG_DIR \
        --data_path $DATA_PATH_IMAGENET \
        --dataset ImageNet16-120 \
        --rand_seed 3
    ((count++))
    echo " $count/$total_runs"
    echo "======================================"

    sleep 10
done

echo " $total_runs "