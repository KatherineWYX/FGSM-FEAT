###!/usr/bin/env bash
##set -e
##
##GPU=5
##OUT=./output
##DATA_DIR=./tiny-imagenet-200
##PY=FGSM_LAW_Tiny_ImageNet.py
##
### 1) sweep lam_scale
##for LS in 0.05 0.1 0.15; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/Tiny_ImageNet/lam_scale_${LS} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale ${LS} \
##    --lam_start 50 \
##    --beta 0.6
##done
##
### 2) sweep lam_start
##for LST in 40 50 60; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/Tiny_ImageNet/lam_start_${LST} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale 0.1 \
##    --lam_start ${LST} \
##    --beta 0.6
##done
##
### 3) sweep beta
##for B in 0.2 0.4 0.6 0.8; do
##  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
##    --out_dir ${OUT}/Tiny_ImageNet/beta_${B} \
##    --data-dir ${DATA_DIR} \
##    --lam_scale 0.1 \
##    --lam_start 50 \
##    --beta ${B}
##done
#
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.05 --data-dir ./tiny-imagenet-200 --lam_scale 0.05 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.10 --data-dir ./tiny-imagenet-200 --lam_scale 0.10 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.15 --data-dir ./tiny-imagenet-200 --lam_scale 0.15 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.30 --data-dir ./tiny-imagenet-200 --lam_scale 0.30 --lam_start 40 --beta 0.2
##
##CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.2 --data-dir ./tiny-imagenet-200 --lam_scale 0.2 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.12 --data-dir ./tiny-imagenet-200 --lam_scale 0.12 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.18 --data-dir ./tiny-imagenet-200 --lam_scale 0.18 --lam_start 40 --beta 0.2
##CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.14 --data-dir ./tiny-imagenet-200 --lam_scale 0.14 --lam_start 40 --beta 0.2
#CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_30_beta_0.2/lam_scale_0.16 --data-dir ./tiny-imagenet-200 --lam_scale 0.16 --lam_start 30 --beta 0.2
#CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_50_beta_0.2/lam_scale_0.16 --data-dir ./tiny-imagenet-200 --lam_scale 0.16 --lam_start 50 --beta 0.2
#CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_60_beta_0.2/lam_scale_0.16 --data-dir ./tiny-imagenet-200 --lam_scale 0.16 --lam_start 60 --beta 0.2
#CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_70_beta_0.2/lam_scale_0.16 --data-dir ./tiny-imagenet-200 --lam_scale 0.16 --lam_start 70 --beta 0.2
#CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2_1.4/lam_scale_0.16 --data-dir ./tiny-imagenet-200 --lam_scale 0.16 --lam_start 40 --beta 0.2
#
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.08 --data-dir ./tiny-imagenet-200 --lam_scale 0.08 --lam_start 40 --beta 0.1
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.09 --data-dir ./tiny-imagenet-200 --lam_scale 0.09 --lam_start 40 --beta 0.1
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.10 --data-dir ./tiny-imagenet-200 --lam_scale 0.10 --lam_start 40 --beta 0.1
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.11 --data-dir ./tiny-imagenet-200 --lam_scale 0.11 --lam_start 40 --beta 0.1
##CUDA_VISIBLE_DEVICES=5 python3 FGSM_LAW_Tiny_ImageNet.py --out_dir ./output/Tiny_ImageNet/lam_start_40_beta_0.2/lam_scale_0.12 --data-dir ./tiny-imagenet-200 --lam_scale 0.12 --lam_start 40 --beta 0.1

##!/usr/bin/env bash
#set -e
#
#GPU=5
#OUT=./output/Tiny_ImageNet_best
#DATA_DIR=./tiny-imagenet-200
#PY=FGSM_LAW_Tiny_ImageNet.py  # 替换成你的训练脚本路径
#
#LAM_SCALE=0.15
#LAM_START=40
#BETA=0.2
#
#for RUN in {1..10}; do
#    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
#        --out_dir ${OUT}/lam_scale_${LAM_SCALE}_lam_start_${LAM_START}_beta_${BETA}/run_${RUN} \
#        --data-dir $DATA_DIR \
#        --lam_scale $LAM_SCALE \
#        --lam_start $LAM_START \
#        --beta $BETA
#done

#!/usr/bin/env bash
set -e

GPU=3
OUT=./output/Tiny_ImageNet
DATA_DIR=tiny-imagenet-200
PY=FGSM_LAW_Tiny_ImageNet.py  # 你的训练脚本

# Tiny ImageNet 建议参数组合
# lamda 初始值: 30, 32, 34
# lam_scale: 0.08, 0.10, 0.12, 0.15
# beta 固定 0.2
# lam_start 固定 40

LAMDA_LIST=(30 32 34)
LAM_SCALE_LIST=(0.08 0.10 0.12 0.15)
BETA=0.2
LAM_START=40
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
