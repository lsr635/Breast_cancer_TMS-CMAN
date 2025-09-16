import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from tms_cman_model import TMS_CMAN
from dataset import BreastDMDataset
from torch.utils.data import DataLoader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        subtraction = batch.get('subtraction')
        if subtraction is not None:
            subtraction = subtraction.to(device)
        outputs = model(frames.unsqueeze(2), subtraction) 
        loss = criterion(outputs['logits'], labels)

        if 'aux_logits' in outputs:
            aux_loss = criterion(outputs['aux_logits'], labels)
            loss = loss + 0.3 * aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            subtraction = batch.get('subtraction')
            if subtraction is not None:
                subtraction = subtraction.to(device)
            
            outputs = model(frames.unsqueeze(2), subtraction)
            loss = criterion(outputs['logits'], labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(outputs['logits'], dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = specificity = 0
    
    return avg_loss, accuracy, auc, sensitivity, specificity

def main():
    parser = argparse.ArgumentParser(description='Train TMS-CMAN on BreastDM Dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory (dataset_exp_1 or dataset_exp_2)')
    parser.add_argument('--dataset_type', type=str, choices=['exp_1', 'exp_2'], required=True,
                       help='Dataset type: exp_1 (no subtraction) or exp_2 (with subtraction)')
    parser.add_argument('--img_size', type=int, default=96,
                       help='Image size (default: 96)')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'full'], help='Model complexity')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"{args.dataset_type}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Save directory: {save_dir}")

    print("Loading datasets...")
    train_dataset = BreastDMDataset(
        root_dir=args.data_dir,
        mode='train',
        img_size=args.img_size,
        dataset_type=args.dataset_type,
        augmentation=True
    )
    
    val_dataset = BreastDMDataset(
        root_dir=args.data_dir,
        mode='val',
        img_size=args.img_size,
        dataset_type=args.dataset_type,
        augmentation=False
    )
    
    test_dataset = BreastDMDataset(
        root_dir=args.data_dir,
        mode='test',
        img_size=args.img_size,
        dataset_type=args.dataset_type,
        augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("Creating model...")
    if args.model_type == 'simple':
        model = TMS_CMAN(
            num_classes=2,
            num_frames=9,
            cnn_backbone='resnet50',  
            transformer_backbone='vit_tiny_patch16_224',
            use_subtraction=(args.dataset_type == 'exp_2'),
            pretrained=True
        )
    else:
        model = TMS_CMAN(
            num_classes=2,
            num_frames=9,
            cnn_backbone='convnext_tiny',
            transformer_backbone='swin_tiny_patch4_window7_224',
            use_subtraction=(args.dataset_type == 'exp_2'),
            pretrained=True
        )
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    best_auc = 0
    best_epoch = 0
    
    print("\nStarting training...")
    print("="*50)
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-"*30)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_sens, val_spec = validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        print(f"        Sensitivity: {val_sens:.4f}, Specificity: {val_spec:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'args': args
            }
            
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"✓ Saved best model (AUC: {best_auc:.4f})")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pth')['model_state_dict'])
    test_loss, test_acc, test_auc, test_sens, test_spec = validate(model, test_loader, criterion, device)
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Sensitivity: {test_sens:.4f}")
    print(f"  Specificity: {test_spec:.4f}")

if __name__ == '__main__':
    main()