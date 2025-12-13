#!/usr/bin/env bash
# =============================================================================
# 最佳参数搜索脚本
# 基于之前实验结果分析，尝试可能产生更好结果的参数组合
# 输出目录: ./output_best/
# =============================================================================
set -e

# 进入项目目录
cd /home/ls/wangyaxian/FGSM-FEAT/_backup

# GPU设置 (可根据需要修改)
GPU=0

# 输出目录
OUT_BASE=../output_best

# 数据目录
CIFAR_DATA=cifar-data
TINY_DATA=./tiny-imagenet-200

# 创建输出目录
mkdir -p ${OUT_BASE}

echo "=============================================="
echo "开始运行最佳参数搜索实验"
echo "输出目录: ${OUT_BASE}"
echo "=============================================="

# =============================================================================
# CIFAR-10 实验
# 当前最佳: beta=0.6, lambda=12, lam_scale=0.1, lam_start=40, PGD=58.40%
# 建议尝试: beta=0.7/0.8, lam_start=30/35
# =============================================================================
echo ""
echo "========== CIFAR-10 实验 =========="

# 实验1: beta=0.7, lam_start=35
echo "[CIFAR10] Running: beta=0.7, lambda=12, scale=0.1, start=35"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.7_start_35 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.7 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 35

# 实验2: beta=0.7, lam_start=40
echo "[CIFAR10] Running: beta=0.7, lambda=12, scale=0.1, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.7_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.7 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 40

# 实验3: beta=0.8, lam_start=40
echo "[CIFAR10] Running: beta=0.8, lambda=12, scale=0.1, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.8_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.8 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 40

# 实验4: beta=0.6, lam_start=30
echo "[CIFAR10] Running: beta=0.6, lambda=12, scale=0.1, start=30"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.6_start_30 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.6 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 30

# 实验5: beta=0.6, lam_start=35
echo "[CIFAR10] Running: beta=0.6, lambda=12, scale=0.1, start=35"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.6_start_35 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.6 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 35

echo "[CIFAR10] 完成 5 个实验"

# =============================================================================
# CIFAR-100 实验
# 当前最佳: beta=0.2, lambda=42, lam_scale=0.01, PGD=31.35%
# 建议尝试: lam_scale=0.005/0.008/0.003, 以及加上lam_start
# =============================================================================
echo ""
echo "========== CIFAR-100 实验 =========="

# 实验1: lam_scale=0.005, 无lam_start
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.005, no_start"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.005_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.005

# 实验2: lam_scale=0.008, 无lam_start
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.008, no_start"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.008_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.008

# 实验3: lam_scale=0.003, 无lam_start
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.003, no_start"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.003_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.003

# 实验4: lam_scale=0.005, lam_start=50
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.005, start=50"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.005_start_50 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.005 \
    --lam_start 50

# 实验5: lam_scale=0.008, lam_start=50
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.008, start=50"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.008_start_50 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.008 \
    --lam_start 50

# 实验6: lam_scale=0.01, lam_start=40 (原最佳加上lam_start)
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.01, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.01_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.01 \
    --lam_start 40

# 实验7: lam_scale=0.01, lam_start=60
echo "[CIFAR100] Running: beta=0.2, lambda=42, scale=0.01, start=60"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.01_start_60 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.01 \
    --lam_start 60

echo "[CIFAR100] 完成 7 个实验"

# =============================================================================
# Tiny_ImageNet 实验
# 当前最佳: beta=0.2, lambda=38, lam_scale=0.16, lam_start=40, PGD=24.39%
# 建议尝试: lam_scale=0.17/0.18/0.2, beta=0.15/0.25
# =============================================================================
echo ""
echo "========== Tiny_ImageNet 实验 =========="

# 实验1: lam_scale=0.17
echo "[TinyImageNet] Running: beta=0.2, lambda=38, scale=0.17, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.17_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.17 \
    --lam_start 40

# 实验2: lam_scale=0.18
echo "[TinyImageNet] Running: beta=0.2, lambda=38, scale=0.18, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.18_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.18 \
    --lam_start 40

# 实验3: lam_scale=0.19
echo "[TinyImageNet] Running: beta=0.2, lambda=38, scale=0.19, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.19_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.19 \
    --lam_start 40

# 实验4: beta=0.15, lam_scale=0.16
echo "[TinyImageNet] Running: beta=0.15, lambda=38, scale=0.16, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.16_beta_0.15 \
    --data-dir ${TINY_DATA} \
    --beta 0.15 \
    --lamda 38 \
    --lam_scale 0.16 \
    --lam_start 40

# 实验5: beta=0.25, lam_scale=0.16
echo "[TinyImageNet] Running: beta=0.25, lambda=38, scale=0.16, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.16_beta_0.25 \
    --data-dir ${TINY_DATA} \
    --beta 0.25 \
    --lamda 38 \
    --lam_scale 0.16 \
    --lam_start 40

# 实验6: beta=0.15, lam_scale=0.18
echo "[TinyImageNet] Running: beta=0.15, lambda=38, scale=0.18, start=40"
CUDA_VISIBLE_DEVICES=$GPU python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.18_beta_0.15 \
    --data-dir ${TINY_DATA} \
    --beta 0.15 \
    --lamda 38 \
    --lam_scale 0.18 \
    --lam_start 40

echo "[TinyImageNet] 完成 6 个实验"

# =============================================================================
# 总结
# =============================================================================
echo ""
echo "=============================================="
echo "所有实验完成!"
echo "=============================================="
echo "CIFAR10:       5 个实验"
echo "CIFAR100:      7 个实验"
echo "Tiny_ImageNet: 6 个实验"
echo "总计:          18 个实验"
echo ""
echo "结果保存在: ${OUT_BASE}/"
echo "=============================================="
