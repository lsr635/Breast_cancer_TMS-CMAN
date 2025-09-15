import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix
import argparse
from pathlib import Path

from tms_cman_model import TMS_CMAN, TMS_CMAN_WithSegmentation

class AsymmetricLoss(nn.Module):
    """非对称损失函数，处理类别不平衡"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, x, y):
        # x: logits, y: targets
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        
        loss = -torch.sum(one_sided_w * (los_pos + los_neg))
        
        return loss.mean()

class DCEMRIDataset(torch.utils.data.Dataset):
    """DCE-MRI数据集"""
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for frame_path in sample['frames']:
            frame = self._load_frame(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        frames = torch.stack(frames)  # [T, C, H, W]

        subtraction = None
        if 'subtraction' in sample:
            subtraction = self._load_frame(sample['subtraction'])
            if self.transform:
                subtraction = self.transform(subtraction)
        
        label = sample['label']
        
        return {
            'frames': frames,
            'subtraction': subtraction,
            'label': label,
            'case_id': sample['case_id']
        }
    
    def _load_frame(self, path):
        pass

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.T_0,
            T_mult=2,
            eta_min=config.min_lr
        )
        self.criterion_cls = AsymmetricLoss()
        self.criterion_seg = nn.CrossEntropyLoss()

        if config.use_ema:
            self.ema = self._create_ema_model()

        self.best_auc = 0
        self.best_epoch = 0
        
    def _create_ema_model(self):
        ema_model = type(self.model)(self.config).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            subtraction = batch.get('subtraction')
            if subtraction is not None:
                subtraction = subtraction.to(self.device)
            outputs = self.model(frames, subtraction)
            loss_cls = self.criterion_cls(outputs['logits'], labels)
            loss_aux = self.criterion_cls(outputs['aux_logits'], labels)
            loss = loss_cls + 0.3 * loss_aux
 
            if 'seg_logits' in outputs:
                seg_targets = batch.get('seg_mask')
                if seg_targets is not None:
                    seg_targets = seg_targets.to(self.device)
                    loss_seg = self.criterion_seg(outputs['seg_logits'], seg_targets)
                    loss = loss + 0.5 * loss_seg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if hasattr(self, 'ema'):
                self._update_ema()

            total_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs['logits']).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        auc = roc_auc_score(targets, predictions)
        
        return avg_loss, auc
    
    def validate(self, val_loader):
        model = self.ema if hasattr(self, 'ema') else self.model
        model.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                subtraction = batch.get('subtraction')
                if subtraction is not None:
                    subtraction = subtraction.to(self.device)

                logits_list = []
                for aug in range(self.config.tta_num):
                    aug_frames = self._apply_tta(frames, aug)
                    outputs = model(aug_frames, subtraction)
                    logits_list.append(outputs['logits'])

                logits = torch.stack(logits_list).mean(dim=0)
                loss = self.criterion_cls(logits, labels)
                total_loss += loss.item()
                predictions.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(targets, predictions)
        predictions_binary = (np.array(predictions) > 0.5).astype(int)
        cm = confusion_matrix(targets, predictions_binary)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        return metrics
    
    def _apply_tta(self, x, aug_idx):
        """测试时增强"""
        if aug_idx == 0:
            return x
        elif aug_idx == 1:
            return torch.flip(x, dims=[-1])  
        elif aug_idx == 2:
            return torch.flip(x, dims=[-2])
        else:
            return x
    
    def _update_ema(self, decay=0.999):
        """更新EMA模型"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss, train_auc = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            print(f"Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")

            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.best_epoch = epoch
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')

            if self.config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'train_auc': train_auc,
                    'val_loss': val_metrics['loss'],
                    'val_auc': val_metrics['auc'],
                    'sensitivity': val_metrics['sensitivity'],
                    'specificity': val_metrics['specificity'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
    
    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict() if hasattr(self, 'ema') else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch
        }, filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tta_num', type=int, default=3)
    parser.add_argument('--T_0', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    config = parser.parse_args()

    if config.use_wandb:
        wandb.init(project='tms-cman', config=config)

    model = TMS_CMAN(
        num_classes=2,
        num_frames=9,
        use_subtraction=True,
        pretrained=True
    )
    
    # 如果需要多任务学习
    # model = TMS_CMAN_WithSegmentation(model)
    
    model = model.cuda()
    train_dataset = DCEMRIDataset(config.data_dir, mode='train')
    val_dataset = DCEMRIDataset(config.data_dir, mode='val')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, config.num_epochs)
    
    print(f"\nTraining completed!")
    print(f"Best AUC: {trainer.best_auc:.4f} at epoch {trainer.best_epoch}")

if __name__ == '__main__':
    main()