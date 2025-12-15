#!/usr/bin/env bash
# ============================================================================
# CIFAR-10 补充实验脚本 (并行版本)
# 用于超参数敏感性分析 - 支持多GPU并行
# ============================================================================
set -e

# 激活 conda 环境
source /mnt/data/cpfs/miniconda3/bin/activate feat

# 配置
GPUS=${1:-"0,1,2,3"}  # 默认使用 GPU 0,1,2,3，可通过参数指定
OUT=./output/cifar10_analysis
DATA_DIR=cifar-data
PY=_backup/FGSM_LAW_CIFAR_10.py

# 将GPU列表转为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo "CIFAR-10 补充实验 (并行版本)"
echo "使用 GPU: ${GPUS} (共 ${NUM_GPUS} 个)"
echo "输出目录: ${OUT}"
echo "=============================================="

# 创建输出目录
mkdir -p ${OUT}
mkdir -p ${OUT}/logs

# 定义所有实验任务
declare -a TASKS=(
    # lamda 分析任务 (固定 beta=0.6, lam_scale=0.1)
    "lamda_analysis/lamda_4|--beta 0.6 --lamda 4 --lam_scale 0.1 --lam_start 40"
    "lamda_analysis/lamda_6|--beta 0.6 --lamda 6 --lam_scale 0.1 --lam_start 40"
    "lamda_analysis/lamda_14|--beta 0.6 --lamda 14 --lam_scale 0.1 --lam_start 40"
    "lamda_analysis/lamda_16|--beta 0.6 --lamda 16 --lam_scale 0.1 --lam_start 40"
    "lamda_analysis/lamda_18|--beta 0.6 --lamda 18 --lam_scale 0.1 --lam_start 40"
    "lamda_analysis/lamda_20|--beta 0.6 --lamda 20 --lam_scale 0.1 --lam_start 40"
    # lam_scale 分析任务 (固定 beta=0.6, lamda=12)
    "scale_analysis/scale_0.02|--beta 0.6 --lamda 12 --lam_scale 0.02 --lam_start 40"
    "scale_analysis/scale_0.04|--beta 0.6 --lamda 12 --lam_scale 0.04 --lam_start 40"
    "scale_analysis/scale_0.06|--beta 0.6 --lamda 12 --lam_scale 0.06 --lam_start 40"
    "scale_analysis/scale_0.14|--beta 0.6 --lamda 12 --lam_scale 0.14 --lam_start 40"
    "scale_analysis/scale_0.16|--beta 0.6 --lamda 12 --lam_scale 0.16 --lam_start 40"
    "scale_analysis/scale_0.18|--beta 0.6 --lamda 12 --lam_scale 0.18 --lam_start 40"
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
    
    # 解析任务
    local OUT_SUBDIR=$(echo $TASK | cut -d'|' -f1)
    local PARAMS=$(echo $TASK | cut -d'|' -f2)
    local TASK_NAME=$(basename $OUT_SUBDIR)
    
    echo "[GPU ${GPU}] 开始: ${TASK_NAME}"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 $PY \
        --out_dir ${OUT}/${OUT_SUBDIR} \
        --data-dir ${DATA_DIR} \
        ${PARAMS} \
        > ${OUT}/logs/${TASK_NAME}.log 2>&1
    
    echo "[GPU ${GPU}] 完成: ${TASK_NAME}"
}

# 并行执行任务
echo "开始并行执行..."
echo ""

TASK_IDX=0
declare -a PIDS=()

while [ $TASK_IDX -lt $NUM_TASKS ]; do
    # 为每个GPU分配一个任务
    for GPU in "${GPU_ARRAY[@]}"; do
        if [ $TASK_IDX -ge $NUM_TASKS ]; then
            break
        fi
        
        TASK="${TASKS[$TASK_IDX]}"
        run_task $GPU "$TASK" &
        PIDS+=($!)
        
        TASK_IDX=$((TASK_IDX + 1))
    done
    
    # 等待当前批次完成
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    PIDS=()
done

echo ""
echo "=============================================="
echo "所有补充实验完成!"
echo "=============================================="
echo ""
echo "下一步: 运行可视化脚本"
echo "  python scripts/visualize_cifar10_analysis.py"
echo ""
