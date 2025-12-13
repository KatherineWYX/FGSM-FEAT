"""
标签处理工具函数
"""
import numpy as np
import torch
import torch.nn.functional as F


def label_smoothing(label: torch.Tensor, factor: float, num_classes: int = 10) -> torch.Tensor:
    """
    标签平滑
    
    Args:
        label: 原始标签
        factor: 平滑因子
        num_classes: 类别数量
    
    Returns:
        平滑后的标签（one-hot形式）
    """
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return torch.tensor(result, device=label.device, dtype=torch.float32)


def label_relaxation(label: torch.Tensor, factor: float, num_classes: int = 10) -> torch.Tensor:
    """
    标签松弛（用于动态标签松弛策略）
    
    Args:
        label: 原始标签
        factor: 松弛因子
        num_classes: 类别数量
    
    Returns:
        松弛后的标签
    """
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return torch.tensor(result, device=label.device, dtype=torch.float32)


def LabelSmoothLoss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    标签平滑损失函数
    
    Args:
        input: 模型输出（logits）
        target: 目标标签（平滑后的soft label）
    
    Returns:
        损失值
    """
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


def CW_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Carlini-Wagner损失函数
    
    Args:
        x: 模型输出
        y: 真实标签
    
    Returns:
        CW损失
    """
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()
