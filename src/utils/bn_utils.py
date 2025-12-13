"""
Batch Normalization 工具函数
"""
import torch
import torch.nn as nn


def check_bn(model: nn.Module) -> bool:
    """
    检查模型是否包含BatchNorm层
    
    Args:
        model: PyTorch模型
    
    Returns:
        是否包含BatchNorm层
    """
    flag = [False]
    
    def _check(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            flag[0] = True
    
    model.apply(_check)
    return flag[0]


def reset_bn(module: nn.Module):
    """
    重置BatchNorm层的running statistics
    
    Args:
        module: PyTorch模块
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module: nn.Module, momenta: dict):
    """获取BatchNorm层的momentum值"""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module: nn.Module, momenta: dict):
    """设置BatchNorm层的momentum值"""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model: nn.Module):
    """
    更新BatchNorm buffers
    
    使用训练数据集的一个epoch来估计buffers的平均值
    
    Args:
        loader: 训练数据加载器
        model: 待更新的模型
    """
    if not check_bn(model):
        return
    
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda m: _get_momenta(m, momenta))
    
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.size(0)
        
        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum
        
        model(input)
        n += b
    
    model.apply(lambda m: _set_momenta(m, momenta))
