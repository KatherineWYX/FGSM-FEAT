#!/usr/bin/env python3
"""
CIFAR-10 超参数敏感性分析可视化脚本

生成三组分析图:
1. beta (β) 分析 - Dynamic Label Relaxation
2. lamda (λ_feat) 分析 - Feature-consistency Regularization 初始权重
3. lam_scale (τ) 分析 - 温度因子

使用方法:
    python scripts/visualize_cifar10_analysis.py
    python scripts/visualize_cifar10_analysis.py --output-dir ./figures
"""

import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置论文级别的图表样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


def parse_log(log_path: str) -> Optional[Dict]:
    """解析日志文件，提取参数和结果"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        # 提取参数
        params = {}
        for line in lines:
            if 'Namespace(' in line:
                match = re.search(r'Namespace\((.*?)\)', line)
                if match:
                    params_str = match.group(1)
                    for param in params_str.split(', '):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            params[key.strip()] = value.strip()
                break
        
        if not params:
            return None
        
        # 提取所有 PGD Acc 和 Test Acc
        pgd_accs = []
        test_accs = []
        
        for i, line in enumerate(lines):
            if 'Test Loss' in line and 'PGD Acc' in line:
                if i + 1 < len(lines):
                    data_line = lines[i + 1]
                    data_line = re.sub(r'\[.*?\]', '', data_line).strip()
                    data_line = data_line.replace('-', '').strip()
                    values = data_line.split()
                    if len(values) >= 4:
                        try:
                            test_acc = float(values[1])
                            pgd_acc = float(values[3])
                            if 0 < test_acc < 1 and 0 < pgd_acc < 1:
                                test_accs.append(test_acc)
                                pgd_accs.append(pgd_acc)
                        except:
                            pass
        
        if not pgd_accs:
            return None
        
        return {
            'beta': float(params.get('beta', 0)),
            'lamda': float(params.get('lamda', 0)),
            'lam_scale': float(params.get('lam_scale', 0)),
            'lam_start': params.get('lam_start', 'N/A'),
            'best_pgd_acc': max(pgd_accs),
            'best_test_acc': max(test_accs),
            'log_path': log_path
        }
    except Exception as e:
        print(f"解析失败 {log_path}: {e}")
        return None


def is_cifar10(path: str) -> bool:
    """判断是否为 CIFAR-10 日志"""
    path_lower = path.lower()
    if 'cifar100' in path_lower:
        return False
    if 'cifar10' in path_lower or 'cifar-10' in path_lower:
        return True
    return False


def collect_results(base_dir: str) -> List[Dict]:
    """收集所有 CIFAR-10 实验结果"""
    results = []
    
    for root, dirs, files in os.walk(base_dir):
        if '_backup' in root:
            continue
        for f in files:
            if f.endswith('.log'):
                full_path = os.path.join(root, f)
                if is_cifar10(full_path):
                    result = parse_log(full_path)
                    if result:
                        results.append(result)
    
    return results


def aggregate_results(results: List[Dict], key: str, 
                     fixed_params: Dict[str, float]) -> Dict[float, Tuple[float, float, float, float]]:
    """
    聚合结果，计算均值和标准差
    
    Args:
        results: 所有实验结果
        key: 要分析的参数名
        fixed_params: 需要固定的其他参数
    
    Returns:
        {param_value: (pgd_mean, pgd_std, test_mean, test_std)}
    """
    # 按参数值分组
    grouped = defaultdict(list)
    
    for r in results:
        # 检查是否满足固定参数条件
        match = True
        for param, value in fixed_params.items():
            if abs(r.get(param, -999) - value) > 0.001:
                match = False
                break
        
        if match:
            param_value = r[key]
            grouped[param_value].append(r)
    
    # 计算均值和标准差
    aggregated = {}
    for param_value, group in grouped.items():
        pgd_accs = [r['best_pgd_acc'] for r in group]
        test_accs = [r['best_test_acc'] for r in group]
        
        aggregated[param_value] = (
            np.mean(pgd_accs),
            np.std(pgd_accs) if len(pgd_accs) > 1 else 0,
            np.mean(test_accs),
            np.std(test_accs) if len(test_accs) > 1 else 0,
        )
    
    return aggregated


def plot_parameter_analysis(
    data: Dict[float, Tuple[float, float, float, float]],
    param_name: str,
    param_symbol: str,
    fixed_params_str: str,
    output_path: str,
    title: Optional[str] = None
):
    """
    绘制单参数分析图
    
    Args:
        data: {param_value: (pgd_mean, pgd_std, test_mean, test_std)}
        param_name: 参数名称
        param_symbol: 参数符号 (LaTeX)
        fixed_params_str: 固定参数描述
        output_path: 输出路径
        title: 图标题
    """
    if not data:
        print(f"警告: {param_name} 没有数据，跳过绘图")
        return
    
    # 排序数据
    sorted_items = sorted(data.items(), key=lambda x: x[0])
    x_values = [item[0] for item in sorted_items]
    pgd_means = [item[1][0] * 100 for item in sorted_items]
    pgd_stds = [item[1][1] * 100 for item in sorted_items]
    test_means = [item[1][2] * 100 for item in sorted_items]
    test_stds = [item[1][3] * 100 for item in sorted_items]
    
    # 创建图形 (更扁的比例)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 绘制 PGD Accuracy (Robust)
    ax.errorbar(x_values, pgd_means, yerr=pgd_stds, 
                fmt='o-', color='#E74C3C', capsize=5, capthick=2,
                label='PGD Acc (Robust)', linewidth=2.5, markersize=10)
    
    # 绘制 Test Accuracy (Clean)
    ax.errorbar(x_values, test_means, yerr=test_stds,
                fmt='s--', color='#3498DB', capsize=5, capthick=2,
                label='Test Acc (Clean)', linewidth=2.5, markersize=10)
    
    # 设置标签
    ax.set_xlabel(f'{param_symbol}', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')
    else:
        ax.set_title(f'Effect of {param_symbol} on CIFAR-10', fontsize=18, fontweight='bold')
    
    # 添加固定参数说明
    ax.text(0.02, 0.98, f'Fixed: {fixed_params_str}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 设置图例
    ax.legend(loc='lower right', framealpha=0.9)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 设置 y 轴范围
    all_values = pgd_means + test_means
    y_min = min(all_values) - 5
    y_max = max(all_values) + 5
    ax.set_ylim(y_min, y_max)
    
    # 设置 x 轴刻度
    ax.set_xticks(x_values)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 超参数敏感性分析可视化')
    parser.add_argument('--base-dir', type=str, 
                       default='/mnt/data/cpfs/wangyaxian/FGSM-FEAT',
                       help='项目根目录')
    parser.add_argument('--output-dir', type=str, 
                       default='./figures/cifar10_analysis',
                       help='图片输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CIFAR-10 超参数敏感性分析")
    print("=" * 60)
    
    # 收集所有结果
    print("\n[1/4] 收集实验结果...")
    results = collect_results(args.base_dir)
    print(f"    共找到 {len(results)} 个有效实验结果")
    
    # ================================================================
    # 图1: beta 分析
    # ================================================================
    print("\n[2/4] 生成 beta (β) 分析图...")
    beta_data = aggregate_results(
        results, 
        key='beta',
        fixed_params={'lamda': 12.0, 'lam_scale': 0.1}
    )
    print(f"    数据点: {sorted(beta_data.keys())}")
    
    plot_parameter_analysis(
        data=beta_data,
        param_name='beta',
        param_symbol=r'$\beta$ (Label Relaxation Factor)',
        fixed_params_str=r'$\lambda_{feat}$=12, $\tau$=0.1',
        output_path=os.path.join(args.output_dir, 'cifar10_beta_analysis.png'),
        title='Effect of β on CIFAR-10 (Dynamic Label Relaxation)'
    )
    
    # ================================================================
    # 图2: lamda 分析
    # ================================================================
    print("\n[3/4] 生成 lamda (λ_feat) 分析图...")
    lamda_data = aggregate_results(
        results,
        key='lamda', 
        fixed_params={'beta': 0.6, 'lam_scale': 0.1}
    )
    print(f"    数据点: {sorted(lamda_data.keys())}")
    
    plot_parameter_analysis(
        data=lamda_data,
        param_name='lamda',
        param_symbol=r'$\lambda_{feat}$ (Regularization Weight)',
        fixed_params_str=r'$\beta$=0.6, $\tau$=0.1',
        output_path=os.path.join(args.output_dir, 'cifar10_lamda_analysis.png'),
        title=r'Effect of $\lambda_{feat}$ on CIFAR-10 (Feature Regularization)'
    )
    
    # ================================================================
    # 图3: lam_scale (τ) 分析
    # ================================================================
    print("\n[4/4] 生成 lam_scale (τ) 分析图...")
    scale_data = aggregate_results(
        results,
        key='lam_scale',
        fixed_params={'beta': 0.6, 'lamda': 12.0}
    )
    print(f"    数据点: {sorted(scale_data.keys())}")
    
    plot_parameter_analysis(
        data=scale_data,
        param_name='lam_scale',
        param_symbol=r'$\tau$ (Temperature Factor)',
        fixed_params_str=r'$\beta$=0.6, $\lambda_{feat}$=12',
        output_path=os.path.join(args.output_dir, 'cifar10_tau_analysis.png'),
        title=r'Effect of $\tau$ on CIFAR-10 (Adaptive Adjustment)'
    )
    
    # ================================================================
    # 生成汇总图 (三合一)
    # ================================================================
    print("\n[额外] 生成汇总图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 子图1: beta
    if beta_data:
        sorted_items = sorted(beta_data.items())
        x = [item[0] for item in sorted_items]
        pgd = [item[1][0] * 100 for item in sorted_items]
        test = [item[1][2] * 100 for item in sorted_items]
        axes[0].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', linewidth=2, markersize=8)
        axes[0].plot(x, test, 's--', color='#3498DB', label='Test Acc', linewidth=2, markersize=8)
        axes[0].set_xlabel(r'$\beta$', fontsize=14)
        axes[0].set_ylabel('Accuracy (%)', fontsize=14)
        axes[0].set_title(r'(a) Effect of $\beta$', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
    
    # 子图2: lamda
    if lamda_data:
        sorted_items = sorted(lamda_data.items())
        x = [item[0] for item in sorted_items]
        pgd = [item[1][0] * 100 for item in sorted_items]
        test = [item[1][2] * 100 for item in sorted_items]
        axes[1].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', linewidth=2, markersize=8)
        axes[1].plot(x, test, 's--', color='#3498DB', label='Test Acc', linewidth=2, markersize=8)
        axes[1].set_xlabel(r'$\lambda_{feat}$', fontsize=14)
        axes[1].set_ylabel('Accuracy (%)', fontsize=14)
        axes[1].set_title(r'(b) Effect of $\lambda_{feat}$', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
    
    # 子图3: tau
    if scale_data:
        sorted_items = sorted(scale_data.items())
        x = [item[0] for item in sorted_items]
        pgd = [item[1][0] * 100 for item in sorted_items]
        test = [item[1][2] * 100 for item in sorted_items]
        axes[2].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', linewidth=2, markersize=8)
        axes[2].plot(x, test, 's--', color='#3498DB', label='Test Acc', linewidth=2, markersize=8)
        axes[2].set_xlabel(r'$\tau$', fontsize=14)
        axes[2].set_ylabel('Accuracy (%)', fontsize=14)
        axes[2].set_title(r'(c) Effect of $\tau$', fontsize=14)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Hyperparameter Sensitivity Analysis on CIFAR-10', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'cifar10_analysis_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 已保存: {os.path.join(args.output_dir, 'cifar10_analysis_combined.png')}")
    
    # ================================================================
    # 打印数据汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)
    
    print("\n【beta 分析数据】")
    print(f"{'beta':>8} {'PGD Acc':>12} {'Test Acc':>12}")
    print("-" * 36)
    for k, v in sorted(beta_data.items()):
        print(f"{k:>8.2f} {v[0]*100:>11.2f}% {v[2]*100:>11.2f}%")
    
    print("\n【lamda 分析数据】")
    print(f"{'lamda':>8} {'PGD Acc':>12} {'Test Acc':>12}")
    print("-" * 36)
    for k, v in sorted(lamda_data.items()):
        print(f"{k:>8.1f} {v[0]*100:>11.2f}% {v[2]*100:>11.2f}%")
    
    print("\n【lam_scale (τ) 分析数据】")
    print(f"{'tau':>8} {'PGD Acc':>12} {'Test Acc':>12}")
    print("-" * 36)
    for k, v in sorted(scale_data.items()):
        print(f"{k:>8.2f} {v[0]*100:>11.2f}% {v[2]*100:>11.2f}%")
    
    print("\n" + "=" * 60)
    print(f"所有图片已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
