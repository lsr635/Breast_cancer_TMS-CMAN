import os
import re
import glob
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

def build_transforms(train: bool = True, img_size: int = 224):
    if train:
        return T.Compose([
            T.Resize([256, 256]),
            T.ColorJitter(),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD, inplace=True)
        ])
    else:
        return T.Compose([
            T.Resize([img_size, img_size]),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD, inplace=True)
        ])

def default_phase_sort_key(name: str) -> int:
    # e.g.: patient123_phase00.png / _post1.png
    m = re.search(r'phase(\d+)|post[_\-]?(\d+)|_t(\d+)', name.lower())
    if m:
        g = [g for g in m.groups() if g is not None]
        if g:
            try:
                return int(g[0])
            except:
                return 0
    return 0

class TMSImageFolder(Dataset):
    """
    兼容 ImageFolder 目录结构：
      root/train/<class>/*.png
      root/val/<class>/*.png
      root/test/<class>/*.png

    两种工作模式：
    - 单帧模式：T=1（默认完全兼容现有数据），每张图像为一个样本
    - 时序模式：根据 patientID 聚合同病例多相位图像，组成一个样本（需要命名可解析 patientID 与相位）
    """
    def __init__(self, root: str, phase: str, img_size: int = 224,
                 temporal: bool = False, max_T: int = 9,
                 transform=None, phase_sort_key=default_phase_sort_key):
        """
        root/phase/class_name/*.png
        temporal=False: 单帧模式
        temporal=True: 时序模式，按 patientID 聚合，最多取 max_T 帧（不足补齐或截断）
        """
        super().__init__()
        self.root = os.path.join(root, phase)
        self.classes = [d for d in os.listdir(self.root)
                        if os.path.isdir(os.path.join(self.root, d)) and not d.startswith('.')]
        self.classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.temporal = temporal
        self.max_T = max_T
        self.transform = transform if transform is not None else build_transforms(train=(phase=='train'), img_size=img_size)
        self.img_size = img_size
        self.phase_sort_key = phase_sort_key

        if not temporal:
            self.samples = []
            for c in self.classes:
                folder = os.path.join(self.root, c)
                for p in glob.glob(os.path.join(folder, '*')):
                    if os.path.isfile(p):
                        self.samples.append((p, self.class_to_idx[c]))
        else:
            # 时序模式：以 patientID 聚合
            # 假设文件名包含 patientID，可用正则从文件名分离 patient 与相位；此处给出一个简易规则
            # 规则：patientID 为去掉最后一个下划线后缀的部分，例如 P001_phase00.png -> patient=P001
            buckets: Dict[Tuple[str,int], List[str]] = {}
            for c in self.classes:
                folder = os.path.join(self.root, c)
                files = [p for p in glob.glob(os.path.join(folder, '*')) if os.path.isfile(p)]

                patient_map: Dict[str, List[str]] = {}
                for p in files:
                    base = os.path.basename(p)
                    patient = re.split(r'[_\-]phase|[_\-]post|[_\-]t\d+', base, maxsplit=1)[0]
                    patient = patient.split('.')[0]
                    patient_map.setdefault(patient, []).append(p)

                for patient, plist in patient_map.items():
                    plist.sort(key=lambda x: self.phase_sort_key(os.path.basename(x)))
                    buckets.setdefault((patient, self.class_to_idx[c]), []).extend(plist)

            # 将每个 (patient,label) 的帧列表裁剪/填充到 max_T
            self.samples = []
            for (patient, label), plist in buckets.items():
                uniq = sorted(list(set(plist)), key=lambda x: self.phase_sort_key(os.path.basename(x)))
                if len(uniq) >= self.max_T:
                    uniq = uniq[:self.max_T]
                else:
                    if len(uniq) > 0:
                        last = uniq[-1]
                        uniq = uniq + [last] * (self.max_T - len(uniq))
                    else:
                        continue
                self.samples.append((uniq, label))

        assert len(self.samples) > 0, f'No samples found under {self.root}'

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx):
        if not self.temporal:
            path, label = self.samples[idx]
            x = self._load_image(path)           # [3,H,W]
            return x, label, os.path.basename(path)
        else:
            paths, label = self.samples[idx]     # List[str] 长度=T
            frames = [self._load_image(p) for p in paths]   # List of [3,H,W]
            x = torch.stack(frames, dim=0)       # [T,3,H,W]
            return x, label, [os.path.basename(p) for p in paths]

def build_tms_loader(root: str, phase: str, batch_size: int,
                     img_size: int = 224, temporal: bool = True, max_T: int = 9,
                     num_workers: int = 4, pin_memory: bool = True, shuffle: bool = False):
    dataset = TMSImageFolder(root=root, phase=phase, img_size=img_size,
                             temporal=temporal, max_T=max_T,
                             transform=build_transforms(train=(phase=='train'), img_size=img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if phase=='train' else False,
                        num_workers=num_workers, pin_memory=pin_memory, drop_last=(phase=='train'))
    return loader, dataset.classes