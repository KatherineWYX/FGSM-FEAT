"""
模型评估模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.tensor_utils import clamp
from ..attacks.pgd import attack_pgd


def evaluate_standard(test_loader, model: nn.Module) -> tuple:
    """
    标准测试准确率评估
    
    Args:
        test_loader: 测试数据加载器
        model: 模型
    
    Returns:
        (test_loss, test_acc)
    """
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    return test_loss / n, test_acc / n


def evaluate_robustness(
    test_loader,
    model: nn.Module,
    epsilon: torch.Tensor,
    std: torch.Tensor,
    upper_limit: torch.Tensor,
    lower_limit: torch.Tensor,
    attack_iters: int = 10,
    restarts: int = 1,
) -> tuple:
    """
    对抗鲁棒性评估（使用PGD攻击）
    
    Args:
        test_loader: 测试数据加载器
        model: 模型
        epsilon: 扰动范围
        std: 标准差
        upper_limit: 上界
        lower_limit: 下界
        attack_iters: 攻击迭代次数
        restarts: 重启次数
    
    Returns:
        (pgd_loss, pgd_acc)
    """
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    
    for X, y in test_loader:
        X, y = X.cuda(), y.cuda()
        
        pgd_delta = attack_pgd(
            model, X, y, epsilon, alpha, attack_iters, restarts,
            lower_limit, upper_limit
        )
        
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    return pgd_loss / n, pgd_acc / n
