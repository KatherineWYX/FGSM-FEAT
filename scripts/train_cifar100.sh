#!/bin/bash
# CIFAR-100 训练脚本

# 激活环境
source /home/ls/miniconda3/bin/activate feat

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 训练
python train.py \
    --config configs/cifar100.yaml \
    --data-dir ./data \
    --out-dir ./output/cifar100 \
    --seed 0
