"""
FGSM-LAW Models Module
统一的模型定义模块，支持CIFAR-10, CIFAR-100, Tiny-ImageNet
"""

from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .feature_resnet import FeatureResNet18, FeatureResNet34, FeatureResNet50
from .preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from .wide_resnet import WideResNet
from .vgg import VGG

__all__ = [
    # Standard ResNet
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    # Feature ResNet (with feature output)
    'FeatureResNet18', 'FeatureResNet34', 'FeatureResNet50',
    # PreAct ResNet
    'PreActResNet18', 'PreActResNet34', 'PreActResNet50',
    # Wide ResNet
    'WideResNet',
    # VGG
    'VGG',
]


def get_model(name: str, num_classes: int = 10, with_feature: bool = False):
    """
    工厂函数：根据名称获取模型
    
    Args:
        name: 模型名称 ('ResNet18', 'WideResNet', 'VGG19', etc.)
        num_classes: 分类数量
        with_feature: 是否返回特征层输出（用于Lipschitz正则化）
    
    Returns:
        model: 神经网络模型
    """
    name = name.lower()
    
    if with_feature:
        if 'resnet18' in name:
            return FeatureResNet18(num_classes=num_classes)
        elif 'resnet34' in name:
            return FeatureResNet34(num_classes=num_classes)
        elif 'resnet50' in name:
            return FeatureResNet50(num_classes=num_classes)
        else:
            raise ValueError(f"Feature model not supported for {name}")
    
    if 'resnet18' in name:
        return ResNet18(num_classes=num_classes)
    elif 'resnet34' in name:
        return ResNet34(num_classes=num_classes)
    elif 'resnet50' in name:
        return ResNet50(num_classes=num_classes)
    elif 'resnet101' in name:
        return ResNet101(num_classes=num_classes)
    elif 'resnet152' in name:
        return ResNet152(num_classes=num_classes)
    elif 'preact' in name and '18' in name:
        return PreActResNet18(num_classes=num_classes)
    elif 'preact' in name and '34' in name:
        return PreActResNet34(num_classes=num_classes)
    elif 'preact' in name and '50' in name:
        return PreActResNet50(num_classes=num_classes)
    elif 'wide' in name:
        return WideResNet(num_classes=num_classes)
    elif 'vgg' in name:
        vgg_type = 'VGG19' if '19' in name else 'VGG16'
        return VGG(vgg_type, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
