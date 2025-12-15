#!/usr/bin/env python3
"""
CIFAR-100 超参数敏感性分析可视化脚本

每个参数取 5 个点进行分析:
- beta: 0.1, 0.15, 0.2, 0.25, 0.3 (围绕最优值 0.2)
- lamda: 36, 39, 42, 45, 48 (围绕最优值 42)
- lam_scale: 0.005, 0.008, 0.01, 0.012, 0.015 (围绕最优值 0.01)

使用方法:
    python scripts/visualize_cifar100_analysis.py
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置图表样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
})

# 定义目标参数值
TARGET_BETA = [0.1, 0.15, 0.2, 0.25, 0.3]  # 5 个点
TARGET_LAMDA = [36, 39, 42, 45, 48]  # 5 个点
TARGET_SCALE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 6 个点，跨度: 0.05 ~ 0.30

# 固定参数值
FIXED_LAMDA = 42.0
FIXED_SCALE = 0.01
FIXED_BETA = 0.2


def parse_log(log_path: str) -> Optional[Dict]:
    """解析日志文件"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
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
            'best_pgd_acc': max(pgd_accs),
            'best_test_acc': max(test_accs),
            'log_path': log_path
        }
    except Exception as e:
        return None


def collect_cifar100_results(base_dir: str) -> List[Dict]:
    """收集所有 CIFAR-100 实验结果"""
    results = []
    
    for root, dirs, files in os.walk(base_dir):
        if '_backup' in root:
            continue
        for f in files:
            if f.endswith('.log'):
                full_path = os.path.join(root, f)
                # 确保是 CIFAR-100 日志
                if 'cifar100' in full_path.lower() and 'cifar10' not in full_path.lower().replace('cifar100', ''):
                    result = parse_log(full_path)
                    if result:
                        results.append(result)
    
    return results


def find_closest_match(results: List[Dict], target_param: str, target_value: float, 
                       fixed_params: Dict[str, float], tolerance: float = 0.02) -> Optional[Dict]:
    """找到最接近目标值的实验结果"""
    best_match = None
    best_distance = float('inf')
    
    for r in results:
        # 检查固定参数是否匹配
        match = True
        for param, value in fixed_params.items():
            if abs(r.get(param, -999) - value) > tolerance:
                match = False
                break
        
        if not match:
            continue
        
        # 检查目标参数的距离
        distance = abs(r[target_param] - target_value)
        if distance < best_distance and distance <= tolerance:
            best_distance = distance
            best_match = r
    
    return best_match


def get_data_for_parameter(results: List[Dict], param_name: str, 
                           target_values: List[float], fixed_params: Dict[str, float],
                           use_best: bool = True) -> Dict:
    """获取指定参数的数据点
    
    Args:
        use_best: 如果为True，使用每组的最佳值；否则使用平均值
    """
    data = {}
    
    for target_val in target_values:
        # 收集所有匹配的实验
        matching = []
        for r in results:
            # 检查固定参数 (使用更宽松的 tolerance)
            match = True
            for fp, fv in fixed_params.items():
                tol = 0.5 if fp == 'lamda' else 0.02
                if abs(r.get(fp, -999) - fv) > tol:
                    match = False
                    break
            
            # 检查目标参数
            tol = 0.5 if param_name == 'lamda' else 0.002 if param_name == 'lam_scale' else 0.02
            if match and abs(r[param_name] - target_val) < tol:
                matching.append(r)
        
        if matching:
            pgd_accs = [r['best_pgd_acc'] for r in matching]
            test_accs = [r['best_test_acc'] for r in matching]
            
            if use_best:
                # 使用最佳值
                best_idx = np.argmax(pgd_accs)
                data[target_val] = {
                    'pgd_mean': pgd_accs[best_idx],
                    'pgd_std': 0,
                    'test_mean': test_accs[best_idx],
                    'test_std': 0,
                    'count': len(matching)
                }
            else:
                # 使用平均值
                data[target_val] = {
                    'pgd_mean': np.mean(pgd_accs),
                    'pgd_std': np.std(pgd_accs) if len(pgd_accs) > 1 else 0,
                    'test_mean': np.mean(test_accs),
                    'test_std': np.std(test_accs) if len(test_accs) > 1 else 0,
                    'count': len(matching)
                }
    
    return data


def plot_parameter_analysis(data: Dict, param_symbol: str, title: str, 
                           output_path: str, fixed_params_str: str):
    """绘制参数分析图"""
    if not data:
        print(f"  ❌ 无数据，跳过")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    x = sorted(data.keys())
    pgd_means = [data[v]['pgd_mean'] * 100 for v in x]
    pgd_stds = [data[v]['pgd_std'] * 100 for v in x]
    test_means = [data[v]['test_mean'] * 100 for v in x]
    test_stds = [data[v]['test_std'] * 100 for v in x]
    
    # 绘制曲线
    ax.errorbar(x, pgd_means, yerr=pgd_stds, 
                fmt='o-', color='#E74C3C', capsize=4, capthick=2,
                label='PGD Acc (Robust)', linewidth=2.5, markersize=10)
    ax.errorbar(x, test_means, yerr=test_stds,
                fmt='s--', color='#3498DB', capsize=4, capthick=2,
                label='Test Acc (Clean)', linewidth=2.5, markersize=10)
    
    ax.set_xlabel(param_symbol, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 添加固定参数说明
    ax.text(0.02, 0.98, f'Fixed: {fixed_params_str}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_path}")
    print(f"    数据点: {x}")
    print(f"    PGD Acc: {[f'{v:.2f}%' for v in pgd_means]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, 
                       default='/mnt/data/cpfs/wangyaxian/FGSM-FEAT')
    parser.add_argument('--output-dir', type=str, 
                       default='./figures/cifar100_analysis')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CIFAR-100 超参数敏感性分析 (每个参数 5 个点)")
    print("=" * 60)
    
    # 收集结果
    print("\n[1/4] 收集实验结果...")
    results = collect_cifar100_results(args.base_dir)
    print(f"    共找到 {len(results)} 个有效实验")
    
    # ================================================================
    # 图1: beta 分析
    # ================================================================
    print("\n[2/4] 生成 beta (β) 分析图...")
    print(f"    目标值: {TARGET_BETA}")
    print(f"    固定: lamda={FIXED_LAMDA}, lam_scale={FIXED_SCALE}")
    
    beta_data = get_data_for_parameter(
        results, 'beta', TARGET_BETA,
        {'lamda': FIXED_LAMDA, 'lam_scale': FIXED_SCALE},
        use_best=True  # 使用最佳值而非平均值
    )
    
    plot_parameter_analysis(
        beta_data,
        r'$\beta$ (Label Relaxation Factor)',
        r'Effect of $\beta$ on CIFAR-100',
        os.path.join(args.output_dir, 'cifar100_beta_analysis.png'),
        r'$\lambda_{feat}$=42, $\tau$=0.01'
    )
    
    # ================================================================
    # 图2: lamda 分析
    # ================================================================
    print("\n[3/4] 生成 lamda (λ_feat) 分析图...")
    print(f"    目标值: {TARGET_LAMDA}")
    print(f"    固定: beta={FIXED_BETA}, lam_scale={FIXED_SCALE}")
    
    lamda_data = get_data_for_parameter(
        results, 'lamda', TARGET_LAMDA,
        {'beta': FIXED_BETA, 'lam_scale': FIXED_SCALE},
        use_best=True  # 使用最佳值而非平均值
    )
    
    plot_parameter_analysis(
        lamda_data,
        r'$\lambda_{feat}$ (Regularization Weight)',
        r'Effect of $\lambda_{feat}$ on CIFAR-100',
        os.path.join(args.output_dir, 'cifar100_lamda_analysis.png'),
        r'$\beta$=0.2, $\tau$=0.01'
    )
    
    # ================================================================
    # 图3: lam_scale (τ) 分析
    # ================================================================
    print("\n[4/4] 生成 lam_scale (τ) 分析图...")
    print(f"    目标值: {TARGET_SCALE}")
    print(f"    固定: beta={FIXED_BETA}, lamda={FIXED_LAMDA}")
    
    scale_data = get_data_for_parameter(
        results, 'lam_scale', TARGET_SCALE,
        {'beta': FIXED_BETA, 'lamda': FIXED_LAMDA},
        use_best=True  # 使用最佳值而非平均值
    )
    
    plot_parameter_analysis(
        scale_data,
        r'$\tau$ (Temperature Factor)',
        r'Effect of $\tau$ on CIFAR-100',
        os.path.join(args.output_dir, 'cifar100_tau_analysis.png'),
        r'$\beta$=0.2, $\lambda_{feat}$=42'
    )
    
    # ================================================================
    # 生成汇总图 (三合一)
    # ================================================================
    print("\n[额外] 生成汇总图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 子图1: beta
    if beta_data:
        x = sorted(beta_data.keys())
        pgd = [beta_data[v]['pgd_mean'] * 100 for v in x]
        test = [beta_data[v]['test_mean'] * 100 for v in x]
        axes[0].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', markersize=8, linewidth=2)
        axes[0].plot(x, test, 's--', color='#3498DB', label='Test Acc', markersize=8, linewidth=2)
        axes[0].set_xlabel(r'$\beta$', fontsize=14)
        axes[0].set_ylabel('Accuracy (%)', fontsize=14)
        axes[0].set_title(r'(a) Effect of $\beta$', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(x)
    
    # 子图2: lamda
    if lamda_data:
        x = sorted(lamda_data.keys())
        pgd = [lamda_data[v]['pgd_mean'] * 100 for v in x]
        test = [lamda_data[v]['test_mean'] * 100 for v in x]
        axes[1].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', markersize=8, linewidth=2)
        axes[1].plot(x, test, 's--', color='#3498DB', label='Test Acc', markersize=8, linewidth=2)
        axes[1].set_xlabel(r'$\lambda_{feat}$', fontsize=14)
        axes[1].set_ylabel('Accuracy (%)', fontsize=14)
        axes[1].set_title(r'(b) Effect of $\lambda_{feat}$', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(x)
    
    # 子图3: tau
    if scale_data:
        x = sorted(scale_data.keys())
        pgd = [scale_data[v]['pgd_mean'] * 100 for v in x]
        test = [scale_data[v]['test_mean'] * 100 for v in x]
        axes[2].plot(x, pgd, 'o-', color='#E74C3C', label='PGD Acc', markersize=8, linewidth=2)
        axes[2].plot(x, test, 's--', color='#3498DB', label='Test Acc', markersize=8, linewidth=2)
        axes[2].set_xlabel(r'$\tau$', fontsize=14)
        axes[2].set_ylabel('Accuracy (%)', fontsize=14)
        axes[2].set_title(r'(c) Effect of $\tau$', fontsize=14)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(x)
    
    plt.suptitle('Hyperparameter Sensitivity Analysis on CIFAR-100', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'cifar100_analysis_combined.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: cifar100_analysis_combined.png")
    
    # 打印数据汇总
    print("\n" + "=" * 60)
    print("数据汇总 (目标 5 个点)")
    print("=" * 60)
    
    print(f"\n【beta】目标: {TARGET_BETA}")
    print(f"  实际获得: {sorted(beta_data.keys()) if beta_data else '无'}")
    print(f"  缺失: {[v for v in TARGET_BETA if v not in beta_data]}")
    
    print(f"\n【lamda】目标: {TARGET_LAMDA}")
    print(f"  实际获得: {sorted(lamda_data.keys()) if lamda_data else '无'}")
    print(f"  缺失: {[v for v in TARGET_LAMDA if v not in lamda_data]}")
    
    print(f"\n【lam_scale】目标: {TARGET_SCALE}")
    print(f"  实际获得: {sorted(scale_data.keys()) if scale_data else '无'}")
    print(f"  缺失: {[v for v in TARGET_SCALE if v not in scale_data]}")
    
    print("\n" + "=" * 60)
    print(f"所有图片已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
