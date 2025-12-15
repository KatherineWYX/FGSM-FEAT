#!/usr/bin/env bash
# ============================================================================
# CIFAR-100 完整分析流程
# 包含: 补充实验 + 可视化生成
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 激活 conda 环境 (使用当前服务器的 base 环境，已有 PyTorch + CUDA)
source /mnt/data/cpfs/miniconda3/bin/activate feat

echo "=============================================="
echo "CIFAR-100 超参数敏感性分析 - 完整流程"
echo "=============================================="
echo ""
echo "项目目录: $PROJECT_DIR"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 解析参数
MODE=${1:-"all"}  # all, exp-only, vis-only
GPUS=${2:-"0,1,2,3"}    # 默认使用 GPU 0,1,2,3

case $MODE in
    "all")
        echo "模式: 完整流程 (实验 + 可视化)"
        ;;
    "exp-only")
        echo "模式: 仅运行实验"
        ;;
    "vis-only")
        echo "模式: 仅生成可视化"
        ;;
    *)
        echo "用法: $0 [all|exp-only|vis-only] [gpu_ids]"
        echo "示例:"
        echo "  $0 all 0,1,2,3    # 使用 4 个 GPU 运行完整流程"
        echo "  $0 exp-only 0     # 仅运行实验，使用 GPU 0"
        echo "  $0 vis-only       # 仅生成可视化"
        exit 1
        ;;
esac

echo ""

# ============================================================================
# 步骤1: 运行补充实验
# ============================================================================
if [[ "$MODE" == "all" || "$MODE" == "exp-only" ]]; then
    echo "=============================================="
    echo "[步骤1] 运行 CIFAR-100 补充实验"
    echo "=============================================="
    echo ""
    echo "补充实验清单 (共 8 个):"
    echo "  - beta: 0.1, 0.25, 0.3 (固定 lamda=42, lam_scale=0.01)"
    echo "  - lamda: 39, 45, 48 (固定 beta=0.2, lam_scale=0.01)"
    echo "  - lam_scale: 0.20, 0.25 (固定 beta=0.2, lamda=42)"
    echo ""
    
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    if [ $NUM_GPUS -gt 1 ]; then
        echo "使用并行版本 (${NUM_GPUS} GPUs: ${GPUS})"
        bash scripts/run_supplement_cifar100_parallel.sh "$GPUS"
    else
        echo "使用串行版本 (GPU: ${GPUS})"
        bash scripts/run_supplement_cifar100.sh "$GPUS"
    fi
    
    echo ""
    echo "✓ 补充实验完成"
    echo "实验结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
fi

# ============================================================================
# 步骤2: 生成可视化
# ============================================================================
if [[ "$MODE" == "all" || "$MODE" == "vis-only" ]]; then
    echo "=============================================="
    echo "[步骤2] 生成可视化图表"
    echo "=============================================="
    echo ""
    echo "目标参数值:"
    echo "  - beta: 0.1, 0.15, 0.2, 0.25, 0.3 (5个点)"
    echo "  - lamda: 36, 39, 42, 45, 48 (5个点)"
    echo "  - lam_scale: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 (6个点)"
    echo ""
    
    python scripts/visualize_cifar100_analysis.py \
        --base-dir "$PROJECT_DIR" \
        --output-dir "$PROJECT_DIR/figures/cifar100_analysis"
    
    echo ""
    echo "✓ 可视化生成完成"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo "=============================================="
echo "全部完成!"
echo "=============================================="
echo ""
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "生成的图片位于:"
echo "  $PROJECT_DIR/figures/cifar100_analysis/"
echo ""
echo "包含文件:"
echo "  - cifar100_beta_analysis.png      (β 分析)"
echo "  - cifar100_lamda_analysis.png     (λ_feat 分析)"
echo "  - cifar100_tau_analysis.png       (τ 分析)"
echo "  - cifar100_analysis_combined.png  (三合一汇总图)"
echo ""

# ============================================================================
# 拉起 stress.py
# ============================================================================
echo "=============================================="
echo "拉起 stress.py..."
echo "=============================================="
cd /mnt/data/cpfs/wangyaxian/model_train
python stress.py --gpus 0,1,2,3
