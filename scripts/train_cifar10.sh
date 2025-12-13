#!/bin/bash
# CIFAR-10 训练脚本

# 激活环境
source /home/ls/miniconda3/bin/activate feat

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 训练
python train.py \
    --config configs/default.yaml \
    --data-dir ./data \
    --out-dir ./output/cifar10 \
    --seed 0
