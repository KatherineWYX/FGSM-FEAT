###!/usr/bin/env bash
##set -e
##
##GPU=5
##OUT=./output
##DATA_DIR=cifar-data
##PY=FGSM_LAW_CIFAR100.py
##
### 1) sweep lam_scale
##for LS in 0.05 0.1 0.15; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/CIFAR100/lam_scale_${LS} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale ${LS} \
##    --lam_start 50 \
##    --beta 0.6
##done
##
### 2) sweep lam_start
##for LST in 40 50 60; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/CIFAR100/lam_start_${LST} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale 0.1 \
##    --lam_start ${LST} \
##    --beta 0.6
##done
##
### 3) sweep beta
##for B in 0.2 0.4 0.6 0.8; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/CIFAR100/beta_${B} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale 0.1 \
##    --lam_start 50 \
##    --beta ${B}
##done
#
##!/bin/bash
#
## CIFAR-100 experiments with different lam_scale values, beta fixed at 0.2
#
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.01 \
##    --data-dir cifar-data \
##    --lam_scale 0.01 \
##    --beta 0.2
##
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.05 \
##    --data-dir cifar-data \
##    --lam_scale 0.05 \
##    --beta 0.2
##
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.08 \
##    --data-dir cifar-data \
##    --lam_scale 0.08 \
##    --beta 0.2
##
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.1 \
##    --data-dir cifar-data \
##    --lam_scale 0.1 \
##    --beta 0.2
##
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.15 \
##    --data-dir cifar-data \
##    --lam_scale 0.15 \
##    --beta 0.2
##
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.3 \
##    --data-dir cifar-data \
##    --lam_scale 0.3 \
##    --beta 0.2
#
#CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
#    --out_dir ./output/CIFAR100/beta_0.4/lam_scale_0.01 \
#    --data-dir cifar-data \
#    --lam_scale 0.01 \
#    --beta 0.4
#
#CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
#    --out_dir ./output/CIFAR100/beta_0.4/lam_scale_0.05 \
#    --data-dir cifar-data \
#    --lam_scale 0.05 \
#    --beta 0.4
#
#CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
#    --out_dir ./output/CIFAR100/beta_0.4/lam_scale_0.08 \
#    --data-dir cifar-data \
#    --lam_scale 0.08 \
#    --beta 0.4
#
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
##    --out_dir ./output/CIFAR100/beta_0.2/lam_scale_0.1 \
##    --data-dir cifar-data \
##    --lam_scale 0.1 \
##    --beta 0.2
#
CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ./output/CIFAR100/beta_0.4/lam_scale_0.15 \
    --data-dir cifar-data \
    --lam_scale 0.15 \
    --beta 0.4
#
#CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_CIFAR100.py \
#    --out_dir ./output/CIFAR100/beta_0.4/lam_scale_0.3 \
#    --data-dir cifar-data \
#    --lam_scale 0.3 \
#    --beta 0.4
#

##!/usr/bin/env bash
#set -e
#
#GPU=5
#OUT=./output/CIFAR100_best
#DATA_DIR=cifar-data
#PY=FGSM_LAW_CIFAR100.py  # 你的训练脚本路径
#
#BETA=0.2
#LAM_SCALE=0.01
#LAM_START=50
#
#for RUN in {1..10}; do
#    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
#        --out_dir ${OUT}/beta_${BETA}_lamscale_${LAM_SCALE}_lamstart_${LAM_START}/run_${RUN} \
#        --data-dir $DATA_DIR \
#        --beta $BETA \
#        --lam_scale $LAM_SCALE \
#        --lam_start $LAM_START
#done

#!/usr/bin/env bash
set -e

GPU=5
OUT=./output/CIFAR100
DATA_DIR=cifar-data
PY=FGSM_LAW_CIFAR100.py  # 你的训练脚本

# CIFAR-100 建议参数组合
# lamda 初始值: 35~38
# lam_scale: 0.01~0.05
# beta 固定 0.2
# lam_start 固定 50

LAMDA_LIST=(35 36 37 38)
LAM_SCALE_LIST=(0.03 0.05 0.08 0.10 0.15)
BETA=0.2
LAM_START=50
REPEAT=3

for LAMDA in "${LAMDA_LIST[@]}"; do
  for LS in "${LAM_SCALE_LIST[@]}"; do
    for RUN in $(seq 1 $REPEAT); do
      CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/lamda_${LAMDA}_lamscale_${LS}_run${RUN} \
        --data-dir ${DATA_DIR} \
        --lamda ${LAMDA} \
        --lam_scale ${LS} \
        --lam_start ${LAM_START} \
        --beta ${BETA}
    done
  done
done

