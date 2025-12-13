"""
PGD (Projected Gradient Descent) 攻击
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.tensor_utils import clamp


def attack_pgd(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    attack_iters: int,
    restarts: int,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
) -> torch.Tensor:
    """
    PGD攻击
    
    Args:
        model: 目标模型
        X: 输入图像
        y: 真实标签
        epsilon: 扰动范围
        alpha: 步长
        attack_iters: 攻击迭代次数
        restarts: 重启次数
        lower_limit: 像素值下界
        upper_limit: 像素值上界
    
    Returns:
        最优扰动
    """
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        
        # 随机初始化
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(
                -epsilon[i][0][0].item(), 
                epsilon[i][0][0].item()
            )
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    
    return max_delta


def evaluate_pgd(
    test_loader,
    model: nn.Module,
    attack_iters: int,
    restarts: int,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    normalize_fn=None,
) -> tuple:
    """
    使用PGD攻击评估模型鲁棒性
    
    Args:
        test_loader: 测试数据加载器
        model: 目标模型
        attack_iters: 攻击迭代次数
        restarts: 重启次数
        epsilon: 扰动范围
        alpha: 步长
        lower_limit: 像素值下界
        upper_limit: 像素值上界
        normalize_fn: 归一化函数（可选）
    
    Returns:
        (pgd_loss, pgd_acc)
    """
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
            if normalize_fn is not None:
                output = model(normalize_fn(X + pgd_delta))
            else:
                output = model(X + pgd_delta)
            
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    return pgd_loss / n, pgd_acc / n
