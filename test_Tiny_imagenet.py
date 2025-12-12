# diagnose_and_test_tiny_imagenet.py
import argparse, os, math, importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

# 你的数据集类
from TinyImageNet import TinyImageNet
from cutout import Cutout

# =========================
# 1) 模型：PreActResNet18（Tiny-ImageNet 版，pool 可调）
# =========================
class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet18_Tiny(nn.Module):
    """64x64 输入；layer4 输出 8x8；pool_kernel=4 -> 2x2 -> fc_in=512*4=2048"""
    def __init__(self, num_classes=200, pool_kernel=4):
        super().__init__()
        self.in_planes = 64
        self.pool_kernel = pool_kernel
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, 2, stride=2)
        fc_in = 512 * (8 // pool_kernel) * (8 // pool_kernel)
        self.linear = nn.Linear(fc_in, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_kernel)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# =========================
# 2) 从 ckpt 推断 head 形状并构建匹配模型
# =========================
def build_model_from_ckpt(ckpt_path, device):
    sd_raw = torch.load(ckpt_path, map_location='cpu')
    sd = sd_raw.get('state_dict', sd_raw)

    # 找到线性层权重
    lin_w = None; lin_key = None
    for k,v in sd.items():
        if k.endswith('linear.weight'):
            lin_w, lin_key = v, k; break
    if lin_w is None:
        for k,v in sd.items():
            if any(x in k for x in ['fc.weight', 'classifier.weight']):
                lin_w, lin_key = v, k; break
    if lin_w is None:
        raise RuntimeError('在 ckpt 中找不到 head 权重（*.linear.weight / fc.weight / classifier.weight）')

    out_features, in_features = lin_w.shape
    # 推断 pool_kernel
    if   in_features == 2048: pool_kernel = 4
    elif in_features == 512:  pool_kernel = 8
    else:
        k = int(round(8 / math.sqrt(in_features/512)))
        pool_kernel = max(1, min(8, k))
        print(f'[WARN] 非常规 fc_in={in_features}，推断 pool_kernel={pool_kernel}')

    print(f'[CKPT] {lin_key} shape={tuple(lin_w.shape)} -> num_classes={out_features}, fc_in={in_features}, pool_kernel={pool_kernel}')
    model = PreActResNet18_Tiny(num_classes=out_features, pool_kernel=pool_kernel).to(device)

    # 清前缀再加载
    cleaned = {k.replace('module.','').replace('model.',''): v for k,v in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f'[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}')
    if missing:   print('  missing(head) sample:', missing[:8])
    if unexpected:print('  unexpected(head) sample:', unexpected[:8])

    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(2,3,64,64, device=device))
    print('[SANITY] forward out shape:', tuple(y.shape))
    return model

# =========================
# 3) Loader（与训练一致：val 仅 ToTensor，无 Normalize）
# =========================
def build_loaders_like_train(root, batch_size, use_cutout=False, n_holes=1, length=6):
    tf_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
    ])
    if use_cutout:
        tf_train.transforms.append(Cutout(n_holes=n_holes, length=length))
    tf_val = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(root, 'train', transform=tf_train, in_memory=True)
    valset   = TinyImageNet(root, 'val',   transform=tf_val,   in_memory=True)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=pin)
    test_loader  = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=pin)
    return train_loader, test_loader

# =========================
# 4) 评测（内置实现 + 可选调用 utils 中同名函数）
# =========================
@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    n,corr,loss_sum = 0,0,0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits,(list,tuple)): logits = logits[0]
        loss = F.cross_entropy(logits, y, reduction='sum')
        loss_sum += loss.item()
        corr += (logits.argmax(1)==y).sum().item()
        n += y.size(0)
    return loss_sum/max(1,n), corr/max(1,n)

def fgsm_eval(model, loader, eps, device):
    model.eval()
    n,corr,loss_sum = 0,0,0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x_adv = x.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        if isinstance(logits,(list,tuple)): logits=logits[0]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + eps * x_adv.grad.sign()
            x_adv.clamp_(0,1)
            out = model(x_adv)
            if isinstance(out,(list,tuple)): out=out[0]
            loss_sum += F.cross_entropy(out, y, reduction='sum').item()
            corr += (out.argmax(1)==y).sum().item()
            n += y.size(0)
    return loss_sum/max(1,n), corr/max(1,n)

def pgd_linf_eval(model, loader, eps, alpha, steps, restarts, device):
    model.eval()
    n,corr,loss_sum = 0,0,0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        # 多重重启取最强
        best_adv = x.clone()
        best_correct = torch.zeros_like(y, dtype=torch.bool)
        for _ in range(restarts):
            delta = torch.empty_like(x).uniform_(-eps, eps)
            x_adv = (x + delta).clamp(0,1)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                logits = model(x_adv)
                if isinstance(logits,(list,tuple)): logits=logits[0]
                loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x+eps), x-eps).clamp(0,1)
            with torch.no_grad():
                preds = model(x_adv)
                if isinstance(preds,(list,tuple)): preds=preds[0]
                correct = (preds.argmax(1)==y)
                # 更新 best（把成功攻破的保留为 x_adv）
                update = ~correct & best_correct
                # 对于之前还正确的，如果这次错误了也更新
                update = ~correct
                best_adv[update] = x_adv[update]
                best_correct[update] = False
        with torch.no_grad():
            out = model(best_adv)
            if isinstance(out,(list,tuple)): out=out[0]
            loss_sum += F.cross_entropy(out, y, reduction='sum').item()
            corr += (out.argmax(1)==y).sum().item()
            n += y.size(0)
    return loss_sum/max(1,n), corr/max(1,n)

def pgd_cw_eval(model, loader, eps, alpha, steps, device, kappa=0.0):
    """PGD with CW loss（无二次替代），untargeted:
       loss = max(max_{i≠y} z_i - z_y, -kappa). 这里用梯度上升（增加 margin）。
    """
    model.eval()
    n,corr,loss_sum = 0,0,0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0,1)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            if isinstance(logits,(list,tuple)): logits=logits[0]
            onehot = torch.zeros_like(logits).scatter_(1, y.view(-1,1), 1)
            correct_logit = (logits*onehot).sum(1, keepdim=True)
            max_other = (logits - 1e9*onehot).max(1, keepdim=True)[0]
            cw_loss = (max_other - correct_logit).clamp(min=-kappa).mean()
            grad = torch.autograd.grad(cw_loss, x_adv)[0]
            # 上升
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x+eps), x-eps).clamp(0,1)
        with torch.no_grad():
            out = model(x_adv)
            if isinstance(out,(list,tuple)): out=out[0]
            # 为了可比，loss 用 CE（也可以输出 cw_loss 的平均）
            loss_sum += F.cross_entropy(out, y, reduction='sum').item()
            corr += (out.argmax(1)==y).sum().item()
            n += y.size(0)
    return loss_sum/max(1,n), corr/max(1,n)

# 自动解析 utils 中是否有同名函数（有则优先用）
def resolve_eval_from_utils():
    fgsm_fn = None; cw_fn = None; std_vec = None
    for mod_name in ['utils', 'utils02']:
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        if fgsm_fn is None:
            for cand in ['evaluate_fgsm','eval_fgsm','fgsm_eval']:
                if hasattr(m, cand) and callable(getattr(m, cand)):
                    fgsm_fn = getattr(m, cand)
                    print(f'[RESOLVE] FGSM from {mod_name}.{cand}')
                    break
        if cw_fn is None:
            for cand in ['evaluate_pgd_cw','evaluate_cw','eval_pgd_cw','evaluate_pgd_CW']:
                if hasattr(m, cand) and callable(getattr(m, cand)):
                    cw_fn = getattr(m, cand)
                    print(f'[RESOLVE] CW from {mod_name}.{cand}')
                    break
        # std（某些工程会提供用于 eps/std 的缩放）
        if std_vec is None and hasattr(m, 'std'):
            try:
                sv = getattr(m, 'std')
                std_vec = sv.view(-1).tolist() if torch.is_tensor(sv) else list(sv)
                print(f'[RESOLVE] std from {mod_name}:', std_vec)
            except Exception:
                pass
    return fgsm_fn, cw_fn, std_vec

# =========================
# 5) 其它工具
# =========================
def has_normalize(tf):
    if tf is None: return False
    if isinstance(tf, transforms.Normalize): return True
    if isinstance(tf, transforms.Compose):
        return any(isinstance(t, transforms.Normalize) for t in tf.transforms)
    return False

def wnid_order(ds):
    if hasattr(ds, 'label_text_to_number') and isinstance(ds.label_text_to_number, dict):
        d = ds.label_text_to_number
        return [wn for wn,_ in sorted(d.items(), key=lambda kv: kv[1])]
    if hasattr(ds, 'label_texts'):
        return list(ds.label_texts)
    return None

@torch.no_grad()
def quick_acc(loader, model, device, n=2000):
    model.eval()
    n_tot=0; corr=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if n_tot + len(x) > n:
            x = x[:n-n_tot]; y = y[:n-n_tot]
        out = model(x)
        if isinstance(out,(list,tuple)): out=out[0]
        corr += (out.argmax(1)==y).sum().item()
        n_tot += len(x)
        if n_tot >= n: break
    return corr/max(1,n_tot)

# =========================
# 6) AutoAttack
# =========================
def run_autoattack(model, test_loader, eps_pixel, norm, bs, n_ex):
    from autoattack import AutoAttack
    device = next(model.parameters()).device
    x_all = torch.cat([x for x,_ in test_loader], 0).to(device)
    y_all = torch.cat([y for _,y in test_loader], 0).to(device)
    print('len(test_loader.dataset)=', len(test_loader.dataset),
          '| x_all.shape=', tuple(x_all.shape), 'y_all.shape=', tuple(y_all.shape))
    aa = AutoAttack(model, norm=norm, eps=eps_pixel, version='standard', log_path=os.path.join('.', 'aa_log.txt'))
    aa.run_standard_evaluation(x_all[:n_ex], y_all[:n_ex], bs=bs)

# =========================
# 7) main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='tiny-imagenet-200 根目录')
    ap.add_argument('--model_path', required=True, help='best_model.pth 路径')
    ap.add_argument('--batch_size', type=int, default=500)
    ap.add_argument('--epsilon', type=int, default=8)   # 像素空间 eps
    ap.add_argument('--norm', type=str, default='Linf') # AutoAttack 用
    ap.add_argument('--n_ex', type=int, default=10000)
    # PGD/CW 参数（内置实现用）
    ap.add_argument('--pgd_alpha', type=float, default=2.0/255.0)
    ap.add_argument('--pgd_steps', type=int, default=10)
    ap.add_argument('--pgd_steps_20', type=int, default=20)
    ap.add_argument('--pgd_steps_50', type=int, default=50)
    ap.add_argument('--pgd_restarts', type=int, default=1)
    ap.add_argument('--cw_alpha', type=float, default=1.0/255.0)
    ap.add_argument('--cw_steps', type=int, default=20)
    ap.add_argument('--cw_kappa', type=float, default=0.0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[DEVICE]', device)

    # 模型
    model = build_model_from_ckpt(args.model_path, device)

    # DataLoader（val 无 Normalize）
    train_loader, test_loader = build_loaders_like_train(args.data_dir, args.batch_size, use_cutout=False)

    # 基本自检
    print('[TEST TF] has Normalize:', has_normalize(getattr(test_loader.dataset, 'transform', None)))
    xb, yb = next(iter(test_loader))
    print('[VAL batch stats] mean=',
          [round(x,4) for x in xb.mean([0,2,3]).tolist()],
          'std=', [round(x,4) for x in xb.std([0,2,3]).tolist()],
          'min=', [round(x,4) for x in xb.amin([0,2,3]).tolist()],
          'max=', [round(x,4) for x in xb.amax([0,2,3]).tolist()],
          )
    tr_order = wnid_order(train_loader.dataset)
    te_order = wnid_order(test_loader.dataset)
    if tr_order and te_order:
        print('num_classes train/val:', len(tr_order), len(te_order), '| identical?', tr_order==te_order)

    # 干净精度（probe + 全量）
    probe = quick_acc(test_loader, model, device, 2000)
    print(f'[PROBE] clean@2k = {probe:.4f}')
    clean_loss, clean_acc = eval_clean(model, test_loader, device)
    print(f'[CLEAN] acc={clean_acc:.4f} loss={clean_loss:.4f}')

    # 解析 utils 中是否有现成的 FGSM/CW
    FGSM_UTIL, CW_UTIL, std_from_utils = resolve_eval_from_utils()
    eps = args.epsilon/255.0

    # FGSM
    if FGSM_UTIL is not None:
        try:
            fgsm_loss, fgsm_acc = FGSM_UTIL(test_loader, model, 1)
            print(f'[FGSM(util)] acc={fgsm_acc:.4f} loss={fgsm_loss:.4f}')
        except Exception as e:
            print('[FGSM(util)] 调用失败，改用内置实现：', repr(e))
            fgsm_loss, fgsm_acc = fgsm_eval(model, test_loader, eps, device)
            print(f'[FGSM(builtin)] acc={fgsm_acc:.4f} loss={fgsm_loss:.4f}')
    else:
        fgsm_loss, fgsm_acc = fgsm_eval(model, test_loader, eps, device)
        print(f'[FGSM(builtin)] acc={fgsm_acc:.4f} loss={fgsm_loss:.4f}')

    # PGD-10/20/50（内置）
    pgd10_loss, pgd10_acc = pgd_linf_eval(model, test_loader, eps, args.pgd_alpha, args.pgd_steps, args.pgd_restarts, device)
    print(f'[PGD-10] acc={pgd10_acc:.4f} loss={pgd10_loss:.4f}')
    pgd20_loss, pgd20_acc = pgd_linf_eval(model, test_loader, eps, args.pgd_alpha, args.pgd_steps_20, args.pgd_restarts, device)
    print(f'[PGD-20] acc={pgd20_acc:.4f} loss={pgd20_loss:.4f}')
    pgd50_loss, pgd50_acc = pgd_linf_eval(model, test_loader, eps, args.pgd_alpha, args.pgd_steps_50, args.pgd_restarts, device)
    print(f'[PGD-50] acc={pgd50_acc:.4f} loss={pgd50_loss:.4f}')

    # CW（优先 utils，失败则内置 PGD-CW）
    if CW_UTIL is not None:
        try:
            cw_loss, cw_acc = CW_UTIL(test_loader, model, args.cw_steps, 1)  # 你的 utils 里通常是 (loader, model, steps, restarts)
            print(f'[CW(util)-{args.cw_steps}] acc={cw_acc:.4f} loss={cw_loss:.4f}')
        except Exception as e:
            print('[CW(util)] 调用失败，改用内置 PGD-CW：', repr(e))
            cw_loss, cw_acc = pgd_cw_eval(model, test_loader, eps, args.cw_alpha, args.cw_steps, device, kappa=args.cw_kappa)
            print(f'[CW(builtin)-{args.cw_steps}] acc={cw_acc:.4f} loss={cw_loss:.4f}')
    else:
        cw_loss, cw_acc = pgd_cw_eval(model, test_loader, eps, args.cw_alpha, args.cw_steps, device, kappa=args.cw_kappa)
        print(f'[CW(builtin)-{args.cw_steps}] acc={cw_acc:.4f} loss={cw_loss:.4f}')

    # AutoAttack（像素空间 eps）
    try:
        from autoattack import AutoAttack
        print(f'[AA] eps={eps} norm={args.norm}')
        run_autoattack(model, test_loader, eps, args.norm, args.batch_size, args.n_ex)
    except Exception as e:
        print('[AA] 运行失败 ->', repr(e))

if __name__ == '__main__':
    main()
