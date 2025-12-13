"""
张量操作工具函数
"""
import torch


def clamp(X: torch.Tensor, lower_limit: torch.Tensor, upper_limit: torch.Tensor) -> torch.Tensor:
    """
    将张量X限制在[lower_limit, upper_limit]范围内
    
    Args:
        X: 输入张量
        lower_limit: 下界
        upper_limit: 上界
    
    Returns:
        裁剪后的张量
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    标准化张量
    
    Args:
        X: 输入张量
        mean: 均值
        std: 标准差
    
    Returns:
        标准化后的张量
    """
    return (X - mean) / std


def get_epsilon_alpha(epsilon: int, alpha: int, std: torch.Tensor):
    """
    获取归一化后的epsilon和alpha
    
    Args:
        epsilon: 原始epsilon (整数, 0-255)
        alpha: 原始alpha (整数, 0-255)
        std: 标准差张量
    
    Returns:
        归一化后的epsilon和alpha
    """
    epsilon_norm = (epsilon / 255.) / std
    alpha_norm = (alpha / 255.) / std
    return epsilon_norm, alpha_norm


def get_limits(mean: torch.Tensor, std: torch.Tensor):
    """
    获取图像的上下界（用于对抗扰动裁剪）
    
    Args:
        mean: 均值
        std: 标准差
    
    Returns:
        upper_limit, lower_limit
    """
    upper_limit = ((1 - mean) / std)
    lower_limit = ((0 - mean) / std)
    return upper_limit, lower_limit
