#!/usr/bin/env python3
"""
FGSM-LAW 训练脚本

使用方法:
    python train.py --config configs/default.yaml
    python train.py --config configs/cifar100.yaml
"""
import argparse
import logging
import os
import sys
import yaml

import numpy as np
import torch

from src.trainers import FGSMLAWTrainer
from src.data import get_dataloaders

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='FGSM-LAW Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.out_dir:
        config['output']['out_dir'] = args.out_dir
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # 设置随机种子
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    logger.info(f"Config: {config}")
    
    # 加载数据
    train_loader, test_loader = get_dataloaders(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        cutout_enabled=config['augmentation']['cutout']['enabled'],
        cutout_n_holes=config['augmentation']['cutout']['n_holes'],
        cutout_length=config['augmentation']['cutout']['length'],
    )
    
    # 创建训练器
    trainer = FGSMLAWTrainer(config)
    
    # 开始训练
    output_dir = config['output']['out_dir']
    trainer.train(train_loader, test_loader, output_dir)


if __name__ == '__main__':
    main()
