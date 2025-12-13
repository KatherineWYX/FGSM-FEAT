"""
指数移动平均 (Exponential Moving Average) 模块
用于自动权重平均策略
"""
import copy
import torch


class EMA:
    """
    EMA (Exponential Moving Average) 模型管理器
    
    用于实现自动权重平均策略，在训练过程中维护模型参数的指数移动平均
    
    Args:
        model: 待跟踪的模型
        alpha: EMA衰减系数 (默认0.999)
        buffer_ema: 是否对buffer也应用EMA (默认True)
    """
    
    def __init__(self, model, alpha: float = 0.999, buffer_ema: bool = True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self._get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def _get_model_state(self):
        """获取模型状态的深拷贝"""
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

    def update_params(self, model):
        """
        更新EMA参数
        
        Args:
            model: 当前训练模型
        """
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )
        
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name] + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        
        self.step += 1

    def apply_shadow(self):
        """将EMA参数应用到模型"""
        self.backup = self._get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """恢复原始参数"""
        self.model.load_state_dict(self.backup)

    def get_ema_model(self):
        """获取EMA模型"""
        return self.model
