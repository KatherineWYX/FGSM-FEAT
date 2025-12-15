#!/usr/bin/env bash
# ============================================================================
# CIFAR-100 补充实验脚本
# 围绕最优参数取 5 个点进行分析
# 最优参数: beta=0.2, lamda=42, lam_scale=0.01
# ============================================================================
set -e

# 激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

# 配置
GPU=${1:-0}  # 默认使用 GPU 0
OUT=./output/cifar100_analysis
DATA_DIR=cifar-data
PY=_backup/FGSM_LAW_CIFAR100.py

# 创建输出目录
mkdir -p ${OUT}
mkdir -p ${OUT}/logs

echo "=============================================="
echo "CIFAR-100 补充实验"
echo "GPU: ${GPU}"
echo "输出目录: ${OUT}"
echo "=============================================="

# ============================================================================
# 实验组1: beta 分析 (固定 lamda=42, lam_scale=0.01)
# 目标取值: 0.1, 0.15, 0.2, 0.25, 0.3
# 现有: 0.15, 0.2
# 需补充: 0.1, 0.25, 0.3
# ============================================================================
echo ""
echo "[实验组1] beta 分析 (固定 lamda=42, lam_scale=0.01)"
echo "补充: beta = 0.1, 0.25, 0.3"
echo "--------------------------------------------"

for BETA in 0.1 0.25 0.3; do
    echo ">>> 运行 beta=${BETA}"
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/beta_analysis/beta_${BETA} \
        --data-dir ${DATA_DIR} \
        --beta ${BETA} \
        --lamda 42 \
        --lam_scale 0.01 \
        2>&1 | tee ${OUT}/logs/beta_${BETA}.log
    echo "<<< beta=${BETA} 完成"
done

# ============================================================================
# 实验组2: lamda 分析 (固定 beta=0.2, lam_scale=0.01)
# 目标取值: 36, 39, 42, 45, 48
# 现有: 36, 42
# 需补充: 39, 45, 48
# ============================================================================
echo ""
echo "[实验组2] lamda 分析 (固定 beta=0.2, lam_scale=0.01)"
echo "补充: lamda = 39, 45, 48"
echo "--------------------------------------------"

for LAMDA in 39 45 48; do
    echo ">>> 运行 lamda=${LAMDA}"
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/lamda_analysis/lamda_${LAMDA} \
        --data-dir ${DATA_DIR} \
        --beta 0.2 \
        --lamda ${LAMDA} \
        --lam_scale 0.01 \
        2>&1 | tee ${OUT}/logs/lamda_${LAMDA}.log
    echo "<<< lamda=${LAMDA} 完成"
done

# ============================================================================
# 实验组3: lam_scale 分析 (固定 beta=0.2, lamda=42)
# 目标取值: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
# 现有: 0.05, 0.10, 0.15, 0.30
# 需补充: 0.20, 0.25
# ============================================================================
echo ""
echo "[实验组3] lam_scale 分析 (固定 beta=0.2, lamda=42)"
echo "补充: lam_scale = 0.20, 0.25"
echo "--------------------------------------------"

for SCALE in 0.20 0.25; do
    echo ">>> 运行 lam_scale=${SCALE}"
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/scale_analysis/scale_${SCALE} \
        --data-dir ${DATA_DIR} \
        --beta 0.2 \
        --lamda 42 \
        --lam_scale ${SCALE} \
        2>&1 | tee ${OUT}/logs/scale_${SCALE}.log
    echo "<<< lam_scale=${SCALE} 完成"
done

echo ""
echo "=============================================="
echo "所有补充实验完成!"
echo "=============================================="
echo ""
echo "补充实验统计:"
echo "  - beta: 3 个 (0.1, 0.25, 0.3)"
echo "  - lamda: 3 个 (39, 45, 48)"
echo "  - lam_scale: 2 个 (0.20, 0.25)"
echo "  共 8 个实验"
echo ""
