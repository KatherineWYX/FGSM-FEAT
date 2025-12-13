#!/usr/bin/env python3
"""
FGSM-LAW 测试脚本

使用方法:
    python test.py --model_path ./output/best_model.pth --config configs/default.yaml
"""
import argparse
import logging
import sys
import yaml
from collections import OrderedDict

import torch

from src.models import get_model
from src.data import get_dataloaders, get_dataset_stats
from src.trainers.evaluator import evaluate_standard, evaluate_robustness
from src.attacks import evaluate_fgsm

# 尝试导入AutoAttack
try:
    from autoattack import AutoAttack
    HAS_AUTOATTACK = True
except ImportError:
    HAS_AUTOATTACK = False

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
    parser = argparse.ArgumentParser(description='FGSM-LAW Testing')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--epsilon', type=int, default=8,
                        help='Perturbation epsilon (0-255)')
    parser.add_argument('--autoattack', action='store_true',
                        help='Run AutoAttack evaluation')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据集信息
    stats = get_dataset_stats(config['data']['dataset'])
    num_classes = stats['num_classes']
    
    # 加载模型
    model = get_model(
        config['model']['name'],
        num_classes=num_classes,
        with_feature=False
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    logger.info(f"Loaded model from {args.model_path}")
    
    # 加载数据
    _, test_loader = get_dataloaders(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        batch_size=args.batch_size,
        num_workers=0,
        cutout_enabled=False,
    )
    
    # 设置扰动参数
    std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1).to(device)
    mean = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1).to(device)
    epsilon = (args.epsilon / 255.) / std
    upper_limit = ((1 - mean) / std)
    lower_limit = ((0 - mean) / std)
    
    # 评估标准准确率
    test_loss, test_acc = evaluate_standard(test_loader, model)
    logger.info(f"Clean Accuracy: {test_acc * 100:.2f}%")
    
    # 评估PGD攻击
    for iters in [10, 20, 50]:
        pgd_loss, pgd_acc = evaluate_robustness(
            test_loader, model, epsilon, std,
            upper_limit, lower_limit,
            attack_iters=iters, restarts=1
        )
        logger.info(f"PGD-{iters} Accuracy: {pgd_acc * 100:.2f}%")
    
    # AutoAttack评估
    if args.autoattack and HAS_AUTOATTACK:
        logger.info("Running AutoAttack...")
        adversary = AutoAttack(
            model, norm='Linf',
            eps=args.epsilon / 255.,
            version='standard'
        )
        
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
        
        adversary.run_standard_evaluation(
            x_test[:10000], y_test[:10000],
            bs=args.batch_size
        )
    elif args.autoattack and not HAS_AUTOATTACK:
        logger.warning("AutoAttack not installed. Skipping...")


if __name__ == '__main__':
    main()
