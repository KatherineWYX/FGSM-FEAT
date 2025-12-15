#!/usr/bin/env bash
# ============================================================================
# CIFAR-10 补充实验脚本
# 用于超参数敏感性分析
# ============================================================================
set -e

# 激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

# 配置
GPU=${1:-0}  # 默认使用 GPU 0，可通过参数指定
OUT=./output/cifar10_analysis
DATA_DIR=cifar-data
PY=_backup/FGSM_LAW_CIFAR_10.py

# 创建输出目录
mkdir -p ${OUT}
mkdir -p ${OUT}/logs

echo "=============================================="
echo "CIFAR-10 补充实验"
echo "GPU: ${GPU}"
echo "输出目录: ${OUT}"
echo "=============================================="

# ============================================================================
# 实验组1: lamda 分析 (固定 beta=0.6, lam_scale=0.1, lam_start=40)
# 现有: 8, 10, 12
# 补充: 4, 6, 14, 16, 18, 20
# ============================================================================
echo ""
echo "[实验组1] lamda 分析 (固定 beta=0.6, lam_scale=0.1)"
echo "--------------------------------------------"

for LAMDA in 4 6 14 16 18 20; do
    echo ">>> 运行 lamda=${LAMDA}"
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/lamda_analysis/lamda_${LAMDA} \
        --data-dir ${DATA_DIR} \
        --beta 0.6 \
        --lamda ${LAMDA} \
        --lam_scale 0.1 \
        --lam_start 40 \
        2>&1 | tee ${OUT}/logs/lamda_${LAMDA}.log
    echo "<<< lamda=${LAMDA} 完成"
done

# ============================================================================
# 实验组2: lam_scale (τ) 分析 (固定 beta=0.6, lamda=12, lam_start=40)
# 现有: 0.08, 0.1, 0.12
# 补充: 0.02, 0.04, 0.06, 0.14, 0.16, 0.18
# ============================================================================
echo ""
echo "[实验组2] lam_scale (τ) 分析 (固定 beta=0.6, lamda=12)"
echo "--------------------------------------------"

for SCALE in 0.02 0.04 0.06 0.14 0.16 0.18; do
    echo ">>> 运行 lam_scale=${SCALE}"
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/scale_analysis/scale_${SCALE} \
        --data-dir ${DATA_DIR} \
        --beta 0.6 \
        --lamda 12 \
        --lam_scale ${SCALE} \
        --lam_start 40 \
        2>&1 | tee ${OUT}/logs/scale_${SCALE}.log
    echo "<<< lam_scale=${SCALE} 完成"
done

echo ""
echo "=============================================="
echo "所有补充实验完成!"
echo "=============================================="
echo ""
echo "下一步: 运行可视化脚本"
echo "  python scripts/visualize_cifar10_analysis.py"
echo ""
