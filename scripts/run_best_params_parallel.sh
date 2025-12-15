#!/usr/bin/env bash
# =============================================================================
# 最佳参数搜索脚本 (4 GPU 并行版本)
# 使用 GPU 0, 1, 2, 3 并行运行18个实验
# 输出目录: ./output_best/
# =============================================================================
set -e

# 激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

# =============================================================================
# 等待 sft_full.py 运行完成
# =============================================================================
SFT_PID=$(pgrep -f "sft_full.py" || echo "")

if [ -n "$SFT_PID" ]; then
    echo "=============================================="
    echo "检测到 sft_full.py 正在运行 (PID: $SFT_PID)"
    echo "等待其完成后自动开始实验..."
    echo "=============================================="
    
    # 等待进程结束
    while kill -0 $SFT_PID 2>/dev/null; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] sft_full.py 仍在运行，等待中..."
        sleep 60  # 每分钟检查一次
    done
    
    echo ""
    echo "=============================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sft_full.py 已完成!"
    echo "开始运行最佳参数搜索实验..."
    echo "=============================================="
    echo ""
else
    echo "未检测到 sft_full.py 进程，直接开始实验..."
fi

# 进入项目目录
cd /mnt/data/cpfs/wangyaxian/FGSM-FEAT/_backup

# 输出目录
OUT_BASE=../output_best

# 数据目录
CIFAR_DATA=cifar-data
TINY_DATA=./tiny-imagenet-200

# 创建输出目录
mkdir -p ${OUT_BASE}
mkdir -p ${OUT_BASE}/CIFAR10
mkdir -p ${OUT_BASE}/CIFAR100
mkdir -p ${OUT_BASE}/Tiny_ImageNet

# 日志目录
LOG_DIR=${OUT_BASE}/logs
mkdir -p ${LOG_DIR}

echo "=============================================="
echo "开始运行最佳参数搜索实验 (4 GPU 并行)"
echo "输出目录: ${OUT_BASE}"
echo "日志目录: ${LOG_DIR}"
echo "=============================================="

# =============================================================================
# GPU 0: CIFAR-10 全部5个实验 (约7.5小时)
# =============================================================================
(
# 子 shell 需要重新激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

echo "[GPU 0] Starting CIFAR-10 experiments..."

# 实验1: beta=0.7, lam_start=35
echo "[GPU 0] CIFAR10: beta=0.7, lambda=12, scale=0.1, start=35"
CUDA_VISIBLE_DEVICES=0 python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.7_start_35 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.7 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 35 \
    2>&1 | tee ${LOG_DIR}/cifar10_beta0.7_start35.log

# 实验2: beta=0.7, lam_start=40
echo "[GPU 0] CIFAR10: beta=0.7, lambda=12, scale=0.1, start=40"
CUDA_VISIBLE_DEVICES=0 python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.7_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.7 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/cifar10_beta0.7_start40.log

# 实验3: beta=0.8, lam_start=40
echo "[GPU 0] CIFAR10: beta=0.8, lambda=12, scale=0.1, start=40"
CUDA_VISIBLE_DEVICES=0 python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.8_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.8 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/cifar10_beta0.8_start40.log

# 实验4: beta=0.6, lam_start=30
echo "[GPU 0] CIFAR10: beta=0.6, lambda=12, scale=0.1, start=30"
CUDA_VISIBLE_DEVICES=0 python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.6_start_30 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.6 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 30 \
    2>&1 | tee ${LOG_DIR}/cifar10_beta0.6_start30.log

# 实验5: beta=0.6, lam_start=35
echo "[GPU 0] CIFAR10: beta=0.6, lambda=12, scale=0.1, start=35"
CUDA_VISIBLE_DEVICES=0 python3 FGSM_LAW_CIFAR_10.py \
    --out_dir ${OUT_BASE}/CIFAR10/beta_0.6_start_35 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.6 \
    --lamda 12 \
    --lam_scale 0.1 \
    --lam_start 35 \
    2>&1 | tee ${LOG_DIR}/cifar10_beta0.6_start35.log

echo "[GPU 0] CIFAR-10 完成!"
) &
PID_GPU0=$!

# =============================================================================
# GPU 1: CIFAR-100 前4个实验 (约6小时)
# =============================================================================
(
echo "[GPU 1] Starting CIFAR-100 experiments (part 1)..."

# 实验1: lam_scale=0.005, 无lam_start
echo "[GPU 1] CIFAR100: beta=0.2, lambda=42, scale=0.005, no_start"
CUDA_VISIBLE_DEVICES=1 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.005_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.005 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.005_nostart.log

# 实验2: lam_scale=0.008, 无lam_start
echo "[GPU 1] CIFAR100: beta=0.2, lambda=42, scale=0.008, no_start"
CUDA_VISIBLE_DEVICES=1 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.008_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.008 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.008_nostart.log

# 实验3: lam_scale=0.003, 无lam_start
echo "[GPU 1] CIFAR100: beta=0.2, lambda=42, scale=0.003, no_start"
CUDA_VISIBLE_DEVICES=1 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.003_no_start \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.003 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.003_nostart.log

# 实验4: lam_scale=0.005, lam_start=50
echo "[GPU 1] CIFAR100: beta=0.2, lambda=42, scale=0.005, start=50"
CUDA_VISIBLE_DEVICES=1 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.005_start_50 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.005 \
    --lam_start 50 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.005_start50.log

echo "[GPU 1] CIFAR-100 (part 1) 完成!"
) &
PID_GPU1=$!

# =============================================================================
# GPU 2: CIFAR-100 后3个实验 + Tiny_ImageNet 前2个实验 (约8-9小时)
# =============================================================================
(
echo "[GPU 2] Starting CIFAR-100 (part 2) + Tiny_ImageNet experiments..."

# CIFAR-100 实验5: lam_scale=0.008, lam_start=50
echo "[GPU 2] CIFAR100: beta=0.2, lambda=42, scale=0.008, start=50"
CUDA_VISIBLE_DEVICES=2 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.008_start_50 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.008 \
    --lam_start 50 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.008_start50.log

# CIFAR-100 实验6: lam_scale=0.01, lam_start=40
echo "[GPU 2] CIFAR100: beta=0.2, lambda=42, scale=0.01, start=40"
CUDA_VISIBLE_DEVICES=2 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.01_start_40 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.01 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.01_start40.log

# CIFAR-100 实验7: lam_scale=0.01, lam_start=60
echo "[GPU 2] CIFAR100: beta=0.2, lambda=42, scale=0.01, start=60"
CUDA_VISIBLE_DEVICES=2 python3 FGSM_LAW_CIFAR100.py \
    --out_dir ${OUT_BASE}/CIFAR100/scale_0.01_start_60 \
    --data-dir ${CIFAR_DATA} \
    --beta 0.2 \
    --lamda 42 \
    --lam_scale 0.01 \
    --lam_start 60 \
    2>&1 | tee ${LOG_DIR}/cifar100_scale0.01_start60.log

# Tiny_ImageNet 实验1: lam_scale=0.17
echo "[GPU 2] TinyImageNet: beta=0.2, lambda=38, scale=0.17, start=40"
CUDA_VISIBLE_DEVICES=2 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.17_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.17 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.17_beta0.2.log

# Tiny_ImageNet 实验2: lam_scale=0.18
echo "[GPU 2] TinyImageNet: beta=0.2, lambda=38, scale=0.18, start=40"
CUDA_VISIBLE_DEVICES=2 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.18_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.18 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.18_beta0.2.log

echo "[GPU 2] All experiments 完成!"
) &
PID_GPU2=$!

# =============================================================================
# GPU 3: Tiny_ImageNet 后4个实验 (约12-16小时)
# =============================================================================
(
echo "[GPU 3] Starting Tiny_ImageNet experiments..."

# Tiny_ImageNet 实验3: lam_scale=0.19
echo "[GPU 3] TinyImageNet: beta=0.2, lambda=38, scale=0.19, start=40"
CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.19_beta_0.2 \
    --data-dir ${TINY_DATA} \
    --beta 0.2 \
    --lamda 38 \
    --lam_scale 0.19 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.19_beta0.2.log

# Tiny_ImageNet 实验4: beta=0.15, lam_scale=0.16
echo "[GPU 3] TinyImageNet: beta=0.15, lambda=38, scale=0.16, start=40"
CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.16_beta_0.15 \
    --data-dir ${TINY_DATA} \
    --beta 0.15 \
    --lamda 38 \
    --lam_scale 0.16 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.16_beta0.15.log

# Tiny_ImageNet 实验5: beta=0.25, lam_scale=0.16
echo "[GPU 3] TinyImageNet: beta=0.25, lambda=38, scale=0.16, start=40"
CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.16_beta_0.25 \
    --data-dir ${TINY_DATA} \
    --beta 0.25 \
    --lamda 38 \
    --lam_scale 0.16 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.16_beta0.25.log

# Tiny_ImageNet 实验6: beta=0.15, lam_scale=0.18
echo "[GPU 3] TinyImageNet: beta=0.15, lambda=38, scale=0.18, start=40"
CUDA_VISIBLE_DEVICES=3 python3 FGSM_LAW_Tiny_ImageNet.py \
    --out_dir ${OUT_BASE}/Tiny_ImageNet/scale_0.18_beta_0.15 \
    --data-dir ${TINY_DATA} \
    --beta 0.15 \
    --lamda 38 \
    --lam_scale 0.18 \
    --lam_start 40 \
    2>&1 | tee ${LOG_DIR}/tiny_scale0.18_beta0.15.log

echo "[GPU 3] Tiny_ImageNet 完成!"
) &
PID_GPU3=$!

# =============================================================================
# 等待所有GPU完成
# =============================================================================
echo ""
echo "=============================================="
echo "所有任务已在后台启动"
echo "GPU 0 PID: $PID_GPU0 (5 个 CIFAR-10 实验)"
echo "GPU 1 PID: $PID_GPU1 (4 个 CIFAR-100 实验)"
echo "GPU 2 PID: $PID_GPU2 (3 个 CIFAR-100 + 2 个 Tiny 实验)"
echo "GPU 3 PID: $PID_GPU3 (4 个 Tiny_ImageNet 实验)"
echo "=============================================="
echo ""
echo "等待所有任务完成..."
echo "可以用以下命令查看日志:"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""

wait $PID_GPU0
echo "[GPU 0] 已完成"
wait $PID_GPU1
echo "[GPU 1] 已完成"
wait $PID_GPU2
echo "[GPU 2] 已完成"
wait $PID_GPU3
echo "[GPU 3] 已完成"

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
echo "日志保存在: ${LOG_DIR}/"
echo "=============================================="
