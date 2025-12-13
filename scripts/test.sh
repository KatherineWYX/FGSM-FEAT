#!/bin/bash
# 模型测试脚本

# 激活环境
source /home/ls/miniconda3/bin/activate feat

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 测试 (请修改 MODEL_PATH 为实际模型路径)
MODEL_PATH=${1:-"./output/cifar10/best_model.pth"}
CONFIG=${2:-"configs/default.yaml"}

python test.py \
    --model_path $MODEL_PATH \
    --config $CONFIG \
    --batch-size 128 \
    --epsilon 8
