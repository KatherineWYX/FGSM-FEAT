# FGSM-LAW: Fast Adversarial Training with Lipschitz Regularization and Auto Weight Averaging

## 📖 简介

本项目实现了论文 **"Revisiting and Exploring Efficient Fast Adversarial Training via LAW: Lipschitz Regularization and Auto Weight Averaging"** 中提出的方法。

FGSM-LAW 是一种高效的快速对抗训练方法，通过以下技术提高模型的对抗鲁棒性：

### 🔑 核心技术

1. **Lipschitz正则化 (Lipschitz Regularization)**
   - 通过约束模型输出相对于输入扰动的变化来限制局部非线性
   - 防止Catastrophic Overfitting（灾难性过拟合）
   - 核心公式：`L_lip = (MSE(out_adv, out_clean) + MSE(feat_adv, feat_clean)) / (MSE(X_adv, X) + 0.125)`

2. **自动权重平均 (Auto Weight Averaging - EMA)**
   - 使用指数移动平均维护模型权重
   - 根据 `adv_acc / clean_acc < threshold` 动态决定是否更新EMA

3. **动态标签松弛 (Dynamic Label Relaxation)**
   - 使用 `tan(1 - epoch/total_epochs) * beta` 动态调整标签松弛因子
   - 在训练早期使用较大的松弛，后期逐渐减小

4. **自适应正则化系数**
   - 根据 `(clean_acc - adv_acc) / clean_acc` 动态调整Lipschitz正则化强度

## 📁 项目结构

```
FGSM-FEAT/
├── configs/                    # 🔧 配置文件（YAML格式）
│   ├── default.yaml           # CIFAR-10默认配置
│   └── cifar100.yaml          # CIFAR-100配置
├── src/                        # 📦 源代码
│   ├── models/                 # 🧠 模型定义
│   │   ├── resnet.py          # 标准ResNet
│   │   ├── feature_resnet.py  # 带特征输出的ResNet
│   │   ├── preact_resnet.py   # PreAct ResNet
│   │   ├── wide_resnet.py     # Wide ResNet
│   │   └── vgg.py             # VGG
│   ├── data/                   # 📊 数据加载
│   │   └── datasets.py        # CIFAR-10/100, Tiny-ImageNet
│   ├── attacks/                # ⚔️ 对抗攻击
│   │   ├── pgd.py             # PGD攻击
│   │   ├── fgsm.py            # FGSM攻击
│   │   └── cw.py              # CW攻击
│   ├── trainers/               # 🏋️ 训练器
│   │   ├── fgsm_law_trainer.py # FGSM-LAW核心训练器
│   │   └── evaluator.py       # 模型评估
│   └── utils/                  # 🔨 工具函数
│       ├── ema.py             # EMA实现
│       ├── label_utils.py     # 标签处理
│       ├── augmentation.py    # Cutout数据增强
│       └── tensor_utils.py    # 张量操作
├── scripts/                    # 🚀 启动脚本
│   ├── train_cifar10.sh
│   ├── train_cifar100.sh
│   └── test.sh
├── autoattack/                 # AutoAttack评估
├── train.py                    # 训练入口
├── test.py                     # 测试入口
├── requirements.txt            # 依赖列表
├── _backup/                    # 原始代码备份
└── README.md
```

## 🚀 快速开始

### 环境配置

```bash
# 激活conda环境
conda activate feat

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# CIFAR-10 训练
python train.py --config configs/default.yaml --data-dir ./data

# CIFAR-100 训练
python train.py --config configs/cifar100.yaml --data-dir ./data

# 使用脚本
bash scripts/train_cifar10.sh
```

### 测试模型

```bash
# 基础评估（Clean + PGD-10/20/50 + CW）
python test.py --model_path ./output/best_model.pth --config configs/default.yaml

# 使用AutoAttack评估
python test.py --model_path ./output/best_model.pth --config configs/default.yaml --autoattack

# 使用脚本
bash scripts/test.sh ./output/best_model.pth
```

## ⚙️ 配置说明

主要配置参数（见 `configs/default.yaml`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `adversarial.epsilon` | 扰动大小 (0-255) | 8 |
| `adversarial.alpha` | 步长 | 8 |
| `fgsm_law.lamda` | Lipschitz正则化系数 | 12.0 |
| `fgsm_law.lam_scale` | 系数缩放因子 | 0.12 |
| `fgsm_law.lam_start` | 开始自适应调整的epoch | 50 |
| `fgsm_law.beta` | 动态标签松弛因子 | 0.5 |
| `fgsm_law.ema_value` | EMA更新阈值 | 0.82 |
| `fgsm_law.batch_m` | 动量更新系数 | 0.75 |
| `augmentation.label_smoothing_factor` | 标签平滑因子 | 0.7 |

## 📊 典型性能

在CIFAR-10数据集上的结果（ResNet-18）：

| 指标 | 准确率 |
|------|--------|
| Clean Acc | ~84% |
| PGD-10 Acc | ~48% |
| PGD-50 Acc | ~46% |
| AutoAttack | ~44% |

## 📜 License

MIT License
