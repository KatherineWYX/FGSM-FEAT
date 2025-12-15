#!/usr/bin/env bash
# ============================================================================
# CIFAR-10 完整分析流程
# 包含: 补充实验 + 可视化生成
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "CIFAR-10 超参数敏感性分析 - 完整流程"
echo "=============================================="
echo ""
echo "项目目录: $PROJECT_DIR"
echo ""

# 解析参数
MODE=${1:-"all"}  # all, exp-only, vis-only
GPUS=${2:-"0"}    # GPU 列表，如 "0,1,2,3"

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
    echo "[步骤1] 运行补充实验"
    echo "=============================================="
    echo ""
    
    # 检查是否使用并行版本
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    if [ $NUM_GPUS -gt 1 ]; then
        echo "使用并行版本 (${NUM_GPUS} GPUs: ${GPUS})"
        bash scripts/run_supplement_cifar10_parallel.sh "$GPUS"
    else
        echo "使用串行版本 (GPU: ${GPUS})"
        bash scripts/run_supplement_cifar10.sh "$GPUS"
    fi
    
    echo ""
    echo "✓ 补充实验完成"
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
    
    python scripts/visualize_cifar10_analysis.py \
        --base-dir "$PROJECT_DIR" \
        --output-dir "$PROJECT_DIR/figures/cifar10_analysis"
    
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
echo "生成的图片位于:"
echo "  $PROJECT_DIR/figures/cifar10_analysis/"
echo ""
echo "包含文件:"
echo "  - cifar10_beta_analysis.png      (β 分析)"
echo "  - cifar10_lamda_analysis.png     (λ_feat 分析)"
echo "  - cifar10_tau_analysis.png       (τ 分析)"
echo "  - cifar10_analysis_combined.png  (三合一汇总图)"
echo ""

