#!/usr/bin/env bash
# ============================================================================
# CIFAR-100 补充实验脚本 (并行版本)
# 围绕最优参数取 5 个点进行分析
# ============================================================================
set -e

# 激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

# 配置
GPUS=${1:-"0,1,2,3"}  # 默认使用 GPU 0,1,2,3
OUT=./output/cifar100_analysis
DATA_DIR=cifar-data
PY=_backup/FGSM_LAW_CIFAR100.py

# 将GPU列表转为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo "CIFAR-100 补充实验 (并行版本)"
echo "使用 GPU: ${GPUS} (共 ${NUM_GPUS} 个)"
echo "输出目录: ${OUT}"
echo "=============================================="

# 创建输出目录
mkdir -p ${OUT}
mkdir -p ${OUT}/logs

# 定义所有实验任务 (共8个)
declare -a TASKS=(
    # beta 分析 (固定 lamda=42, lam_scale=0.01)
    "beta_analysis/beta_0.1|--beta 0.1 --lamda 42 --lam_scale 0.01"
    "beta_analysis/beta_0.25|--beta 0.25 --lamda 42 --lam_scale 0.01"
    "beta_analysis/beta_0.3|--beta 0.3 --lamda 42 --lam_scale 0.01"
    # lamda 分析 (固定 beta=0.2, lam_scale=0.01)
    "lamda_analysis/lamda_39|--beta 0.2 --lamda 39 --lam_scale 0.01"
    "lamda_analysis/lamda_45|--beta 0.2 --lamda 45 --lam_scale 0.01"
    "lamda_analysis/lamda_48|--beta 0.2 --lamda 48 --lam_scale 0.01"
    # lam_scale 分析 (固定 beta=0.2, lamda=42)
    "scale_analysis/scale_0.20|--beta 0.2 --lamda 42 --lam_scale 0.20"
    "scale_analysis/scale_0.25|--beta 0.2 --lamda 42 --lam_scale 0.25"
)

NUM_TASKS=${#TASKS[@]}
echo "总共 ${NUM_TASKS} 个实验任务"
echo ""

# 运行单个任务的函数
run_task() {
    local GPU=$1
    local TASK=$2
    
    # 子进程需要重新激活 conda 环境
    source /mnt/data/cpfs/miniconda3/bin/activate feat
    
    local OUT_SUBDIR=$(echo $TASK | cut -d'|' -f1)
    local PARAMS=$(echo $TASK | cut -d'|' -f2)
    local TASK_NAME=$(basename $OUT_SUBDIR)
    
    echo "[GPU ${GPU}] 开始: ${TASK_NAME}"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/${OUT_SUBDIR} \
        --data-dir ${DATA_DIR} \
        ${PARAMS} \
        2>&1 | tee ${OUT}/logs/${TASK_NAME}.log
    
    echo "[GPU ${GPU}] 完成: ${TASK_NAME}"
}

# 并行执行任务
echo "开始并行执行..."
echo ""

TASK_IDX=0
declare -a PIDS=()

while [ $TASK_IDX -lt $NUM_TASKS ]; do
    for GPU in "${GPU_ARRAY[@]}"; do
        if [ $TASK_IDX -ge $NUM_TASKS ]; then
            break
        fi
        
        TASK="${TASKS[$TASK_IDX]}"
        run_task $GPU "$TASK" &
        PIDS+=($!)
        
        TASK_IDX=$((TASK_IDX + 1))
    done
    
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    PIDS=()
done

echo ""
echo "=============================================="
echo "所有补充实验完成!"
echo "=============================================="
