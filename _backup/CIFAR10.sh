##!/usr/bin/env bash
#set -e
#
#GPU=5
#OUT=./output
#DATA_DIR=cifar-data
#PY=FGSM_LAW_CIFAR_10.py
#
## 1) sweep lam_scale (lam_start/beta 固定为默认)
#for LS in 0.05 0.1 0.15; do
#  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
#    --out_dir ${OUT}/CIFAR10/lam_scale_${LS} \
#    --data-dir ${DATA_DIR} \
#    --lam_scale ${LS} \
#    --lam_start 50 \
#    --beta 0.6
#done
#
## 2) sweep lam_start (lam_scale/beta 固定为默认)
#for LST in 40 50 60; do
#  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
#    --out_dir ${OUT}/CIFAR10/lam_start_${LST} \
#    --data-dir ${DATA_DIR} \
#    --lam_scale 0.1 \
#    --lam_start ${LST} \
#    --beta 0.6
#done
#
## 3) sweep beta (lam_scale/lam_start 固定为默认)
#for B in 0.2 0.4 0.6 0.8; do
#  CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
#    --out_dir ${OUT}/CIFAR10/beta_${B} \
#    --data-dir ${DATA_DIR} \
#    --lam_scale 0.1 \
#    --lam_start 50 \
#    --beta ${B}
#done

#!/usr/bin/env bash
set -e

GPU=5
OUT=./output
DATA_DIR=cifar-data
PY=FGSM_LAW_CIFAR_10.py

# beta 和 lam_scale 的交叉组合测试
for BETA in 0.4 0.5 0.6; do
  for LS in 0.08 0.1 0.12; do
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
      --out_dir ${OUT}/CIFAR10/beta_${BETA}/lam_scale_${LS} \
      --data-dir ${DATA_DIR} \
      --beta ${BETA} \
      --lam_scale ${LS} \
      --lam_start 40
  done
done
