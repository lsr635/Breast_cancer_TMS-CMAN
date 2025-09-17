import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from datasets_tms import build_tms_loader
from tms_models import TMS_CMAN

def set_seed(seed=8):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate(model, loader, device, num_classes=2):
    model.eval()
    total=0; correct=0; losses=[]
    all_probs=[]; all_labels=[]
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, labels, _ in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = ce(logits, labels)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            total += labels.size(0)
            correct += (pred==labels).sum().item()
            losses.append(loss.item())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
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
            tp = cm[i,i]
            fn = cm[i,:].sum() - tp
            fp = cm[:,i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens_list.append(tp/(tp+fn+1e-8))
            spec_list.append(tn/(tn+fp+1e-8))
        sensitivity = float(np.mean(sens_list))
        specificity = float(np.mean(spec_list))

    return acc, loss, auc, sensitivity, specificity, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/root/autodl-tmp/dataset_exp_1', help='dataset root')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--temporal', type=int, default=0, help='0:单帧兼容; 1:时序模式')
    parser.add_argument('--T', type=int, default=9, help='最大时相数（temporal=1时有效）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=8)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    train_loader, classes = build_tms_loader(
        root=args.path, phase='train', batch_size=args.batch_size, img_size=args.img_size,
        temporal=bool(args.temporal), max_T=args.T, num_workers=4, pin_memory=True, shuffle=True
    )
    val_loader, _ = build_tms_loader(
        root=args.path, phase='val', batch_size=args.batch_size, img_size=args.img_size,
        temporal=bool(args.temporal), max_T=args.T, num_workers=4, pin_memory=True, shuffle=False
    )
    test_loader, _ = build_tms_loader(
        root=args.path, phase='test', batch_size=args.batch_size, img_size=args.img_size,
        temporal=bool(args.temporal), max_T=args.T, num_workers=4, pin_memory=True, shuffle=False
    )
    print(f'Classes: {classes}')

    model = TMS_CMAN(num_classes=args.num_classes, img_size=args.img_size, pretrained=True, T_max=args.T)
    model = nn.DataParallel(model).to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)
        running_loss=0.0; seen=0; correct=0
        for data, labels, _ in pbar:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                b = labels.size(0)
                seen += b
                correct += (pred==labels).sum().item()
                running_loss += loss.item()*b
                pbar.set_postfix(loss=f'{running_loss/seen:.4f}', acc=f'{100.0*correct/seen:.2f}%')

        scheduler.step()

        with torch.no_grad():
            val_acc, val_loss, val_auc, sens, spec, cm = evaluate(model, val_loader, device, args.num_classes)
        print(f'Val  | Loss:{val_loss:.4f} Acc:{val_acc:.2f}% AUC:{val_auc:.4f} Sens:{sens:.4f} Spec:{spec:.4f}')

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs('model/tms_cman', exist_ok=True)
            torch.save(model.module.state_dict(), f'model/tms_cman/tms_cman_best.pth')
            print('✓ Best model updated.')

    with torch.no_grad():
        te_acc, te_loss, te_auc, sens, spec, cm = evaluate(model, test_loader, device, args.num_classes)
    print('Test | Loss:{:.4f} Acc:{:.2f}% AUC:{:.4f} Sens:{:.4f} Spec:{:.4f}'.format(te_loss, te_acc, te_auc, sens, spec))
    print('Confusion Matrix:\n', cm)

if __name__ == '__main__':
    main()