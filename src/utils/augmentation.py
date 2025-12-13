"""
数据增强模块
"""
import torch
import numpy as np


class Cutout:
    """
    Cutout数据增强
    
    随机遮挡图像中的一个或多个方形区域
    
    Reference:
    [1] Terrance DeVries, Graham W. Taylor
        Improved Regularization of Convolutional Neural Networks with Cutout.
        arXiv:1708.04552
    
    Args:
        n_holes: 要遮挡的方形区域数量
        length: 每个方形区域的边长（像素）
    """
    
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: 图像张量，形状为 (C, H, W)
        
        Returns:
            遮挡后的图像
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class AutoAugment:
    """
    AutoAugment策略
    
    Reference:
    [1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le
        AutoAugment: Learning Augmentation Policies from Data. arXiv:1805.09501
    """
    
    def __init__(self, policy: str = 'cifar10'):
        self.policy = policy
        # 具体实现可以从autoaugment.py导入
        
    def __call__(self, img):
        # 简化实现，完整实现见autoaugment.py
        return img
