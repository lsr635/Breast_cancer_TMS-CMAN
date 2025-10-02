import os
import argparse
import copy
import math
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
from torch.utils.data import DataLoader

from datasets import TMSSequenceFolder as TMSImageFolder, default_build_transforms as build_transforms
from models import TMS_CMAN

import pandas as pd

# ---------------- 连续帧输入 灰度输入适配：任意通道 -> 3通道 ----------------
class InputAdapter(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _apply_conv4d(self, x):
        # x: [B,C,H,W]
        if isinstance(self.proj, nn.Identity):
            return x
        if x.shape[1] != self.in_channels:
            raise RuntimeError(f'InputAdapter expects {self.in_channels} channels, but got {x.shape[1]}')
        return self.proj(x)

    def forward(self, x):
        # x: [B,3,H,W] 或 [B,T,3,H,W] 或 [B,C,H,W]
        if x.dim() == 4:
            return self._apply_conv4d(x)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self._apply_conv4d(x)
            x = x.view(B, T, self.out_channels, H, W)
            return x
        else:
            raise RuntimeError(f'InputAdapter expects 4D/5D input, but got {x.dim()}D')

# ---------------- 实用工具 ----------------
def count_full(loader):
    cnt = Counter()
    for _, y, _ in loader:
        cnt.update(y.numpy().tolist())
    return cnt

def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
            self.alpha_is_scalar = True
        else:
            self.alpha = alpha
            self.alpha_is_scalar = False
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt).pow(self.gamma) * logpt
        if self.alpha is not None:
            if self.alpha_is_scalar:
                loss = self.alpha.to(logits.device) * loss
            else:
                at = self.alpha[targets]
                loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        if device is not None:
            self.ema.to(device)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * d + msd[k] * (1.0 - d))

@torch.no_grad()
def evaluate(model, loader, device, num_classes=2, tta=False):
    model.eval()
    total=0; correct=0; losses=[]
    all_probs=[]; all_labels=[]
    ce_eval = nn.CrossEntropyLoss()
    for data, labels, _ in loader:
        data = data.to(device)
        labels = labels.to(device)
        if not tta:
            logits = model(data)
            loss = ce_eval(logits, labels)
            probs = torch.softmax(logits, dim=1)
        else:
            logits1 = model(data)
            logits2 = model(torch.flip(data, dims=[-1]))
            logits = (logits1 + logits2) / 2.0
            loss = ce_eval(logits, labels)
            probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()
        losses.append(loss.item())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = 100.0 * correct / total
    loss = float(np.mean(losses))

    if num_classes==2:
        auc = roc_auc_score(all_labels, all_probs[:,1])
        cm = confusion_matrix(all_labels, all_probs.argmax(axis=1))
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
    else:
        y_onehot = np.eye(num_classes)[all_labels]
        auc = roc_auc_score(y_onehot, all_probs, multi_class='ovr')
        cm = confusion_matrix(all_labels, all_probs.argmax(axis=1))
        sens_list=[]; spec_list=[]
        for i in range(num_classes):
            tp = cm[i,i]; fn = cm[i,:].sum() - tp; fp = cm[:,i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens_list.append(tp/(tp+fn+1e-8))
            spec_list.append(tn/(tn+fp+1e-8))
        sensitivity = float(np.mean(sens_list))
        specificity = float(np.mean(spec_list))

    vals, cnts = np.unique(all_probs.argmax(axis=1), return_counts=True)
    print('[DEBUG] Pred distribution:', dict(zip(vals.tolist(), cnts.tolist())))
    vals, cnts = np.unique(all_labels, return_counts=True)
    print('[DEBUG] True distribution:', dict(zip(vals.tolist(), cnts.tolist())))

    return acc, loss, auc, sensitivity, specificity, cm, all_labels, all_probs[:,1]

@torch.no_grad()
def evaluate_collect(model, loader, device, tta=False):
    """仅收集标签与正类概率，供阈值扫描使用。"""
    model.eval()
    all_labels=[]; all_scores=[]
    for data, labels, _ in loader:
        data = data.to(device)
        if not tta:
            logits = model(data)
        else:
            logits1 = model(data)
            logits2 = model(torch.flip(data, dims=[-1]))
            logits = (logits1 + logits2) / 2.0
        probs = torch.softmax(logits, dim=1)
        all_scores.append(probs[:,1].detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_scores = np.nan_to_num(all_scores, nan=0.5, posinf=1.0, neginf=0.0)
    return all_labels, all_scores

def cls_metrics_from_scores(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    acc = 100.0 * (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return acc, sens, spec, cm

def sweep_threshold(y_true, y_score, step=0.001):
    # 扫描阈值寻找最大 Youden’s J
    best = {'thr':0.5, 'J':-1, 'acc':0, 'sens':0, 'spec':0, 'cm':None}
    thr_values = np.arange(0.0, 1.0+1e-12, step, dtype=np.float64)
    for thr in thr_values:
        acc, sens, spec, cm = cls_metrics_from_scores(y_true, y_score, thr)
        J = sens + spec - 1.0
        if J > best['J']:
            best = {'thr':float(thr), 'J':float(J), 'acc':float(acc),
                    'sens':float(sens), 'spec':float(spec), 'cm':cm}
    return best

def set_bn_eval(module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()

def set_bn_train(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()

# ---------- 参数名匹配工具 ----------
def is_head_param(name: str):
    return any(k in name for k in [
        'cnn_proj', 'tr_proj',
        'tse_', 'tla_', 'cmgca',
        'mil', 'head', 'encoder', 'cls_token', 'norm',
        'input_adapter'
    ])

def is_cnn_backbone(name: str):
    return name.startswith('module.net.cnn.') or name.startswith('module.cnn.')

def is_swin_backbone(name: str):
    return name.startswith('module.net.tr.features.') or name.startswith('module.tr.features.')

def in_cnn_stages(name: str, stages: list):
    if 'input_adapter' in name:
        return False
    return any(f'.cnn.features.{s}.' in name for s in stages)

def in_swin_stages(name: str, stages: list):
    return any(f'.tr.features.{s}.' in name for s in stages)

# ---------- 分阶段解冻 ----------
def apply_stage_freeze_plan(model, plan):
    head_on = plan['unfreeze'].get('head', True)
    cnn_conf = plan['unfreeze'].get('cnn', [])
    swin_conf = plan['unfreeze'].get('swin', [])

    for n, p in model.named_parameters():
        if is_head_param(n):
            p.requires_grad_(head_on)
        elif is_cnn_backbone(n):
            if cnn_conf == 'all':
                p.requires_grad_(True)
            elif isinstance(cnn_conf, list) and len(cnn_conf) > 0:
                p.requires_grad_(in_cnn_stages(n, cnn_conf))
            else:
                p.requires_grad_(False)
        elif is_swin_backbone(n):
            if swin_conf == 'all':
                p.requires_grad_(True)
            elif isinstance(swin_conf, list) and len(swin_conf) > 0:
                p.requires_grad_(in_swin_stages(n, swin_conf))
            else:
                p.requires_grad_(False)
        else:
            p.requires_grad_(head_on)

    if plan.get('bn_eval', False):
        model.apply(set_bn_eval)
    else:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()

def build_optimizer(model, lr_head, lr_cnn, lr_swin, weight_decay):
    head_params = []
    cnn_params = []
    swin_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_head_param(n):
            head_params.append(p)
        elif is_cnn_backbone(n):
            cnn_params.append(p)
        elif is_swin_backbone(n):
            swin_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay})
    if len(cnn_params) > 0:
        param_groups.append({'params': cnn_params, 'lr': lr_cnn, 'weight_decay': weight_decay})
    if len(swin_params) > 0:
        param_groups.append({'params': swin_params, 'lr': lr_swin, 'weight_decay': weight_decay})

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=max(1, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - max(1, warmup_epochs)), eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_epochs)])
    return scheduler

# ---------------- MixUp / CutMix ----------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    new_x = x.clone()
    new_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def apply_mixup_or_cutmix(x, y, do_mixup, do_cutmix, alpha):
    # x: [B,C,H,W] 或 [B,T,C,H,W]
    is_temporal = (x.dim() == 5)
    if not (do_mixup or do_cutmix) or alpha <= 0:
        return x, y, y, 1.0

    if is_temporal:
        B, T, C, H, W = x.shape
        x_ = x.view(B, T*C, H, W)  # 将所有帧并到通道，保证同一 bbox/混合系数作用于所有帧一致位置
    else:
        x_ = x

    if do_cutmix:
        x_aug, y_a, y_b, lam = cutmix_data(x_, y, alpha=alpha)
    else:
        x_aug, y_a, y_b, lam = mixup_data(x_, y, alpha=alpha)

    if is_temporal:
        x_aug = x_aug.view(B, T, C, H, W)
    return x_aug, y_a, y_b, lam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/root/autodl-tmp/dataset_exp_1', help='dataset root')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--temporal', type=int, default=0, help='0:单帧; 1:时序')
    parser.add_argument('--T', type=int, default=9)
    parser.add_argument('--in-channels', type=int, default=1, help='输入通道数（灰度=1，RGB=3）')
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=8)

    parser.add_argument('--use_focal', type=int, default=0, help='1=Focal, 0=CE+LS')
    parser.add_argument('--focal_gamma', type=float, default=1.2)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--ema', type=int, default=1)
    parser.add_argument('--amp', type=int, default=1)
    parser.add_argument('--clip', type=float, default=0.3)

    parser.add_argument('--mixup', type=int, default=1)
    parser.add_argument('--cutmix', type=int, default=1)
    parser.add_argument('--mix_alpha', type=float, default=0.3)
    parser.add_argument('--aug_prob', type=float, default=0.3)

    parser.add_argument('--tta', type=int, default=0)
    parser.add_argument('--swa', type=int, default=0)
    parser.add_argument('--swa_start_ratio', type=float, default=0.8)

    parser.add_argument('--excel', type=str, default='./train_metrics.xlsx')
    parser.add_argument('--stage_warmup', type=int, default=4)
    parser.add_argument('--patience', type=int, default=50)

    # 阈值扫描步长（默认0.001；如要更快可设为 0.005/0.01）
    parser.add_argument('--thr-step', type=float, default=0.001)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # 数据集与 loader
    train_dataset = TMSImageFolder(root=args.path, phase='train', img_size=args.img_size,
                                   temporal=bool(args.temporal), max_T=args.T,
                                   transform=build_transforms(train=True, img_size=args.img_size))
    val_dataset = TMSImageFolder(root=args.path, phase='val', img_size=args.img_size,
                                 temporal=bool(args.temporal), max_T=args.T,
                                 transform=build_transforms(train=False, img_size=args.img_size))
    test_dataset = TMSImageFolder(root=args.path, phase='test', img_size=args.img_size,
                                  temporal=bool(args.temporal), max_T=args.T,
                                  transform=build_transforms(train=False, img_size=args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    classes = train_dataset.classes
    print(f'Classes: {classes}')

    # 类别权重
    train_cnt = count_full(DataLoader(train_dataset, batch_size=256, shuffle=False))
    print('[INFO] Full train class counts:', train_cnt)
    total = sum(train_cnt.values())
    class_weight = []
    for i in range(args.num_classes):
        fi = train_cnt.get(i, 1)
        class_weight.append(total / fi)
    mean_w = sum(class_weight)/len(class_weight)
    class_weight = [w/mean_w for w in class_weight]
    class_weight_t = torch.tensor(class_weight, dtype=torch.float32).to(device)
    print('[INFO] Class weights (mean=1):', class_weight)

    tmp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    tmp_data, _, _ = next(iter(tmp_loader))
    # 兼容时序与单帧
    if tmp_data.dim() == 4:      # [B=1, C, H, W] 或 [C,H,W]（不太可能）
        actual_in_ch = tmp_data.shape[1]
    elif tmp_data.dim() == 5:    # [B=1, T, C, H, W]
        actual_in_ch = tmp_data.shape[2]  # 每帧的通道数，应为3
    else:
        raise RuntimeError(f'Unexpected input dim: {tmp_data.shape}')
    if actual_in_ch != args.in_channels:
        print(f'[INFO] Detected per-frame channels = {actual_in_ch}, override --in-channels {args.in_channels} -> {actual_in_ch}')
        args.in_channels = int(actual_in_ch)

    # 模型：InputAdapter + TMS_CMAN
    input_adapter = InputAdapter(in_channels=args.in_channels, out_channels=3)
    base_model = TMS_CMAN(num_classes=args.num_classes, img_size=args.img_size, pretrained=True, T_max=args.T)
    backbone = nn.Sequential()
    backbone.add_module('input_adapter', input_adapter)
    backbone.add_module('net', base_model)
    model = nn.DataParallel(backbone).to(device)

    # 损失函数
    if args.use_focal == 1:
        alpha_cfg = float(args.focal_alpha)
        print(f'[INFO] Using FocalLoss(gamma={args.focal_gamma}, alpha={alpha_cfg})')
        criterion = FocalLoss(alpha=alpha_cfg, gamma=args.focal_gamma)
    else:
        print('[INFO] Using CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)')
        criterion = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=0.05)

    # AMP
    use_amp = bool(args.amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 阶段计划
    stage_plan = [
        {'name': 'head_only',  'epochs': 8, 'unfreeze': {'head': True, 'cnn': [], 'swin': []},        'bn_eval': True,  'lr_head': 2e-4,  'lr_cnn': 0.0,   'lr_swin': 0.0},
        {'name': 'cnn_last',   'epochs': 6, 'unfreeze': {'head': True, 'cnn': [5,6,7], 'swin': []},   'bn_eval': True,  'lr_head': 1.2e-4,'lr_cnn': 8e-6, 'lr_swin': 0.0},
        {'name': 'cnn_all',    'epochs': 6, 'unfreeze': {'head': True, 'cnn': 'all', 'swin': []},     'bn_eval': True,  'lr_head': 8e-5,  'lr_cnn': 5e-6, 'lr_swin': 0.0},
        {'name': 'swin_last',  'epochs': 6, 'unfreeze': {'head': True, 'cnn': 'all', 'swin': [5,6,7]},'bn_eval': True,  'lr_head': 6e-5,  'lr_cnn': 3e-6, 'lr_swin': 3e-6},
        {'name': 'all_finetune','epochs':10, 'unfreeze': {'head': True, 'cnn': 'all', 'swin': 'all'},  'bn_eval': False, 'lr_head': 3e-5,  'lr_cnn': 2e-6, 'lr_swin': 2e-6},
    ]

    total_stage_epochs = sum(s['epochs'] for s in stage_plan)
    if args.epochs != total_stage_epochs:
        print(f'[WARN] args.epochs({args.epochs}) != stage_plan total({total_stage_epochs}), 将按 stage_plan 训练 {total_stage_epochs} 轮。')

    ema = ModelEMA(model, decay=0.999, device=device) if args.ema else None

    # 记录文件路径（WRITE-EVERY-EPOCH）
    os.makedirs('logs', exist_ok=True)
    excel_path = args.excel
    if not excel_path.endswith('.xlsx'):
        excel_path = excel_path + '.xlsx'
    excel_full = os.path.join('logs', excel_path)
    csv_full = os.path.splitext(excel_full)[0] + '.csv'

    # 早停/SWA
    swa_model = None
    swa_start_epoch = int(total_stage_epochs * args.swa_start_ratio) if args.swa else None
    if args.swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)

    records = []
    best_auc = 0.0
    best_acc = 0.0
    best_auc_epoch = -1
    no_improve = 0
    best_thr = 0.5  # 在验证集上学习得到的最优阈值（Youden J 最大）

    def flush_metrics_to_files(records_list):
        df = pd.DataFrame(records_list)
        os.makedirs(os.path.dirname(excel_full), exist_ok=True)
        df.to_csv(csv_full, index=False)
        try:
            df.to_excel(excel_full, index=False)
        except Exception as e:
            print(f'[WARN] to_excel failed ({e}); CSV 已保存至: {csv_full}')
        else:
            print(f'✓ Metrics saved (epoch rolling) to: {excel_full} and {csv_full}')

    current_epoch = 0
    for stage in stage_plan:
        stage_name = stage['name']
        stage_epochs = stage['epochs']
        print(f'\n===== Stage: {stage_name} | epochs: {stage_epochs} =====')

        # 应用阶段解冻策略
        apply_stage_freeze_plan(model, stage)
        # 优化器/调度器
        optimizer = build_optimizer(model, stage['lr_head'], stage['lr_cnn'], stage['lr_swin'], args.weight_decay)
        scheduler = build_scheduler(optimizer, warmup_epochs=args.stage_warmup, total_epochs=stage_epochs, eta_min=1e-6)

        for e in range(stage_epochs):
            epoch = current_epoch + 1
            model.train()
            if stage_name in ['cnn_all', 'swin_last'] and e == 2:
                set_bn_train(model)

            pbar = tqdm(train_loader, desc=f'[{stage_name}] Epoch {epoch}', ncols=100)
            running_loss=0.0; seen=0; correct=0

            for data, labels, _ in pbar:
                data = data.to(device)
                labels = labels.to(device)

                use_aug = (bool(args.mixup) or bool(args.cutmix)) and (random.random() < args.aug_prob)
                do_cutmix = bool(args.cutmix) and (random.random() < 0.5)
                do_mixup  = bool(args.mixup) and not do_cutmix

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast('cuda', enabled=True):
                        if use_aug and (do_mixup or do_cutmix):
                            data_aug, y_a, y_b, lam = apply_mixup_or_cutmix(data, labels, do_mixup, do_cutmix, args.mix_alpha)
                            logits = model(data_aug)
                            loss = mix_criterion(criterion, logits, y_a, y_b, lam)
                        else:
                            logits = model(data)
                            loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if use_aug and (do_mixup or do_cutmix):
                        data_aug, y_a, y_b, lam = apply_mixup_or_cutmix(data, labels, do_mixup, do_cutmix, args.mix_alpha)
                        logits = model(data_aug)
                        loss = mix_criterion(criterion, logits, y_a, y_b, lam)
                    else:
                        logits = model(data)
                        loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                    optimizer.step()

                if ema is not None:
                    ema.update(model)

                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1)
                    b = labels.size(0)
                    seen += b
                    correct += (pred==labels).sum().item()
                    running_loss += loss.item()*b
                    pbar.set_postfix(loss=f'{running_loss/seen:.4f}', acc=f'{100.0*correct/seen:.2f}%')

            scheduler.step()

            # 验证（先得到常规 AUC 等，再阈值扫描）
            eval_model = ema.ema if ema is not None else model
            val_acc, val_loss, val_auc, sens, spec, cm, y_val, s_val = evaluate(eval_model, val_loader, device, num_classes=args.num_classes, tta=bool(args.tta))
            best_on_val = sweep_threshold(y_val, s_val, step=args.thr_step)
            best_thr = best_on_val['thr']  # 更新全局最佳阈值
            print(f'[VAL-THR] best_thr={best_thr:.3f} | J={best_on_val["J"]:.4f} | acc={best_on_val["acc"]:.2f}% sens={best_on_val["sens"]:.4f} spec={best_on_val["spec"]:.4f}')

            # 记录指标（包含阈值扫描结果）
            train_loss = running_loss/seen if seen>0 else float('nan')
            train_acc = 100.0*correct/seen if seen>0 else float('nan')
            rec = {
                'epoch': epoch,
                'stage': stage_name,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_sens': sens,
                'val_spec': spec,
                'val_best_thr': best_thr,
                'val_bestJ': best_on_val['J'],
                'val_best_acc': best_on_val['acc'],
                'val_best_sens': best_on_val['sens'],
                'val_best_spec': best_on_val['spec']
            }
            records.append(rec)
            print(f'Val  | Loss:{val_loss:.4f} Acc:{val_acc:.2f}% AUC:{val_auc:.4f} Sens:{sens:.4f} Spec:{spec:.4f}')

            # 保存最佳（仍以 AUC 为准）
            to_save = ema.ema if ema is not None else model
            if val_auc > best_auc:
                best_auc = val_auc
                best_auc_epoch = epoch
                no_improve = 0
                os.makedirs('model/tms_cman', exist_ok=True)
                torch.save(to_save.module.state_dict(), f'model/tms_cman/tms_cman_best_auc.pth')
                print('✓ Best AUC model updated.')
            else:
                no_improve += 1

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs('model/tms_cman', exist_ok=True)
                torch.save(to_save.module.state_dict(), f'model/tms_cman/tms_cman_best_acc.pth')
                print('✓ Best Acc model updated.')

            # 每轮写日志
            flush_metrics_to_files(records)

            # 早停
            if args.patience > 0 and no_improve >= args.patience:
                print(f'[Early Stop] No AUC improvement for {args.patience} epochs (last best @ {best_auc_epoch}).')
                current_epoch += 1
                break

            current_epoch += 1

        if args.patience > 0 and no_improve >= args.patience:
            break

    # 最终测试（使用 EMA 模型，阈值采用验证集 best_thr）
    eval_model = ema.ema if ema is not None else model
    te_acc_05, te_loss, te_auc, te_sens_05, te_spec_05, cm_05, y_test, s_test = evaluate(eval_model, test_loader, device, num_classes=args.num_classes, tta=bool(args.tta))
    # 基于 best_thr 的测试指标
    te_acc_thr, te_sens_thr, te_spec_thr, cm_thr = cls_metrics_from_scores(y_test, s_test, best_thr)

    print('Test | (thr=0.5)  Loss:{:.4f} Acc:{:.2f}% AUC:{:.4f} Sens:{:.4f} Spec:{:.4f}'.format(te_loss, te_acc_05, te_auc, te_sens_05, te_spec_05))
    print('Confusion Matrix (thr=0.5):\n', cm_05)
    print('Test | (thr={:.3f}) Acc:{:.2f}% Sens:{:.4f} Spec:{:.4f}'.format(best_thr, te_acc_thr, te_sens_thr, te_spec_thr))
    print('Confusion Matrix (thr={:.3f}):\n'.format(best_thr), cm_thr)

    # 记录最终测试并再次写入（含两套阈值的结果）
    records.append({
        'epoch': current_epoch,
        'stage': 'final_test',
        'train_loss': np.nan,
        'train_acc': np.nan,
        'val_loss': np.nan,
        'val_acc': np.nan,
        'val_auc': np.nan,
        'val_sens': np.nan,
        'val_spec': np.nan,
        'val_best_thr': best_thr,
        'test_loss': te_loss,
        'test_auc': te_auc,
        'test_acc_thr05': te_acc_05,
        'test_sens_thr05': te_sens_05,
        'test_spec_thr05': te_spec_05,
        'test_acc_bestthr': te_acc_thr,
        'test_sens_bestthr': te_sens_thr,
        'test_spec_bestthr': te_spec_thr
    })
    flush_metrics_to_files(records)
    print(f'✓ Final metrics saved to: {excel_full} and {csv_full}')

if __name__ == '__main__':
    main()