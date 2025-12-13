"""
FGSM-LAW 训练器

实现快速对抗训练，包含：
1. Lipschitz正则化
2. 自动权重平均 (EMA)
3. 动态标签松弛
4. 自适应正则化系数
"""
import math
import logging
import os
import time
from typing import Optional, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils.tensor_utils import clamp, get_limits, get_epsilon_alpha
from ..utils.label_utils import label_smoothing, label_relaxation, LabelSmoothLoss
from ..utils.ema import EMA
from ..models import get_model

logger = logging.getLogger(__name__)


class FGSMLAWTrainer:
    """
    FGSM-LAW 训练器
    
    实现论文中的快速对抗训练方法，包含Lipschitz正则化和自动权重平均
    
    Args:
        config: 配置字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据集相关
        self.num_classes = config['model']['num_classes']
        
        # 对抗训练参数
        self.epsilon = config['adversarial']['epsilon']
        self.alpha = config['adversarial']['alpha']
        
        # FGSM-LAW 参数
        self.lamda = config['fgsm_law']['lamda']
        self.lam_scale = config['fgsm_law']['lam_scale']
        self.lam_max = config['fgsm_law']['lam_max']
        self.lam_start = config['fgsm_law']['lam_start']
        self.beta = config['fgsm_law']['beta']
        self.inner_gamma = config['fgsm_law']['inner_gamma']
        self.ema_value = config['fgsm_law']['ema_value']
        self.batch_m = config['fgsm_law']['batch_m']
        
        # 数据增强参数
        self.label_smooth_factor = config['augmentation']['label_smoothing_factor']
        
        # 设置均值和标准差
        self._setup_normalization()
        
        # 初始化模型
        self.model = None
        self.teacher_model = None
        
    def _setup_normalization(self):
        """设置归一化参数"""
        # 使用原始像素值
        self.mean = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1).to(self.device)
        self.upper_limit, self.lower_limit = get_limits(self.mean, self.std)
        
        # 归一化epsilon和alpha
        self.epsilon_norm = (self.epsilon / 255.) / self.std
        self.alpha_norm = (self.alpha / 255.) / self.std
        
    def _build_model(self):
        """构建模型"""
        model = get_model(
            self.config['model']['name'],
            num_classes=self.num_classes,
            with_feature=self.config['model'].get('with_feature', True)
        )
        return model.to(self.device)
    
    def _init_momentum(self, batch_size: int, img_size: int = 32):
        """初始化动量"""
        momentum = torch.zeros(batch_size, 3, img_size, img_size).to(self.device)
        for j in range(len(self.epsilon_norm)):
            momentum[:, j, :, :].uniform_(
                -self.epsilon_norm[j][0][0].item(),
                self.epsilon_norm[j][0][0].item()
            )
        momentum = clamp(
            self.alpha_norm * torch.sign(momentum),
            -self.epsilon_norm,
            self.epsilon_norm
        )
        return momentum
    
    def _compute_lipschitz_loss(
        self,
        output: torch.Tensor,
        fea_output: torch.Tensor,
        adv_output: torch.Tensor,
        ori_fea_output: torch.Tensor,
        X_adv: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算Lipschitz正则化损失
        
        核心公式:
        L_lip = MSE(output, adv_output) + MSE(fea_output, ori_fea_output)
              / (MSE(X_adv, X) + 0.125)
        
        这限制了模型输出相对于输入扰动的变化，提高局部平滑性
        """
        loss_fn = nn.MSELoss(reduction='mean')
        
        numerator = (
            loss_fn(output.float(), adv_output.float()) +
            loss_fn(fea_output.float(), ori_fea_output.float())
        )
        denominator = loss_fn(X_adv.float(), X.float()) + 0.125
        
        return numerator / denominator
    
    def train_epoch(
        self,
        train_loader,
        model: nn.Module,
        teacher_model: EMA,
        optimizer,
        scheduler,
        epoch: int,
        momentum: torch.Tensor,
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            包含训练统计信息的字典
        """
        model.train()
        teacher_model.model.eval()
        
        train_loss = 0
        train_acc = 0
        train_n = 0
        clean_acc_list = []
        adv_acc_list = []
        
        # 动态标签松弛
        inner_gammas = math.tan(1 - (epoch / self.config['training']['epochs'])) * self.beta
        if inner_gammas < self.inner_gamma:
            inner_gammas = self.inner_gamma
        
        batch_size = self.config['data']['batch_size']
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            
            if X.shape[0] != batch_size:
                continue
            
            delta = momentum
            relaxation_label = label_relaxation(y, inner_gammas, self.num_classes)
            
            # ==================== 第一步：生成对抗样本 ====================
            delta.requires_grad = True
            adv_output, ori_fea_output = model(X + delta)
            adv_output = F.softmax(adv_output, dim=1)
            ori_fea_output = F.softmax(ori_fea_output, dim=1)
            
            loss = F.cross_entropy(adv_output, relaxation_label.float())
            
            # 获取干净样本输出
            clean_output, clean_fea = model(X)
            clean_output = F.softmax(clean_output, dim=1)
            clean_fea = F.softmax(clean_fea, dim=1)
            
            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            
            # 更新扰动
            delta.data = clamp(
                delta + self.alpha_norm * torch.sign(grad),
                -self.epsilon_norm,
                self.epsilon_norm
            )
            delta.data = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta = delta.detach()
            
            # 更新动量
            momentum = self.batch_m * momentum + (1.0 - self.batch_m) * delta
            momentum = clamp(momentum, -self.epsilon_norm, self.epsilon_norm)
            momentum = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            
            # ==================== 第二步：计算最终损失 ====================
            X_adv = X + delta[:X.size(0)]
            ori_output, fea_output = model(X_adv)
            output = F.softmax(ori_output, dim=1)
            fea_output = F.softmax(fea_output, dim=1)
            
            # 标签平滑
            label_smooth = label_smoothing(y, self.label_smooth_factor, self.num_classes)
            
            # 总损失 = 分类损失 + Lipschitz正则化
            lipschitz_loss = self._compute_lipschitz_loss(
                output, fea_output, clean_output, clean_fea, X_adv, X
            )
            loss = LabelSmoothLoss(ori_output, label_smooth) + self.lamda * lipschitz_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * y.size(0)
            train_acc += (ori_output.max(1)[1] == y).sum().item()
            adv_acc = (output.max(1)[1] == y).sum().item()
            clean_acc = (clean_output.max(1)[1] == y).sum().item()
            clean_acc_list.append(clean_acc)
            adv_acc_list.append(adv_acc)
            train_n += y.size(0)
            
            # ==================== 自动权重平均 ====================
            if adv_acc / max(clean_acc, 1) < self.ema_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()
            
            scheduler.step()
        
        # 自适应调整正则化系数
        clean_acc_epoch = sum(clean_acc_list) / len(clean_acc_list)
        adv_acc_epoch = sum(adv_acc_list) / len(adv_acc_list)
        
        if epoch > self.lam_start:
            ratio = (clean_acc_epoch - adv_acc_epoch) / max(clean_acc_epoch, 1)
            self.lamda = self.lamda * torch.exp(
                torch.tensor(self.lam_scale * ratio, device=self.device)
            ).item()
            self.lamda = min(max(self.lamda, self.config['fgsm_law']['lamda']),
                           self.config['fgsm_law']['lamda'] * 1.5)
        
        return {
            'train_loss': train_loss / train_n,
            'train_acc': train_acc / train_n,
            'clean_acc': clean_acc_epoch / batch_size,
            'adv_acc': adv_acc_epoch / batch_size,
            'lamda': self.lamda,
        }, momentum
    
    def train(self, train_loader, test_loader, output_dir: str):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型
        self.model = self._build_model()
        self.model.train()
        self.teacher_model = EMA(self.model, alpha=self.config['fgsm_law']['ema_alpha'])
        
        # 优化器和调度器
        epochs = self.config['training']['epochs']
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['lr_max'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        lr_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(lr_steps * 109 / 120), int(lr_steps * 114 / 120)],
            gamma=0.1
        )
        
        # 初始化动量
        batch_size = self.config['data']['batch_size']
        momentum = self._init_momentum(batch_size)
        
        best_pgd_acc = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            stats, momentum = self.train_epoch(
                train_loader, self.model, self.teacher_model,
                optimizer, scheduler, epoch, momentum
            )
            
            epoch_time = time.time() - epoch_start
            
            # 评估
            from .evaluator import evaluate_standard, evaluate_robustness
            test_model = get_model(
                self.config['model']['name'],
                num_classes=self.num_classes,
                with_feature=False
            ).to(self.device)
            test_model.load_state_dict(self.teacher_model.model.state_dict(), strict=False)
            test_model.eval()
            
            test_loss, test_acc = evaluate_standard(test_loader, test_model)
            pgd_loss, pgd_acc = evaluate_robustness(
                test_loader, test_model,
                self.epsilon_norm, self.std,
                self.upper_limit, self.lower_limit
            )
            
            # 日志
            logger.info(
                f"Epoch {epoch}: Time={epoch_time:.1f}s, "
                f"Train Loss={stats['train_loss']:.4f}, Train Acc={stats['train_acc']:.4f}, "
                f"Test Acc={test_acc:.4f}, PGD Acc={pgd_acc:.4f}, Lambda={stats['lamda']:.2f}"
            )
            
            # 保存最佳模型
            if pgd_acc > best_pgd_acc:
                best_pgd_acc = pgd_acc
                torch.save(
                    test_model.state_dict(),
                    os.path.join(output_dir, 'best_model.pth')
                )
        
        # 保存最终模型
        torch.save(
            test_model.state_dict(),
            os.path.join(output_dir, 'final_model.pth')
        )
        
        logger.info(f"Training complete. Best PGD Acc: {best_pgd_acc:.4f}")
