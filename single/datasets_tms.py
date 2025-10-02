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
    Compatible with an ImageFolder layout:
      root/train/<class>/*.png
      root/val/<class>/*.png
      root/test/<class>/*.png

    Supports two modes:
    - Single-frame mode: T=1, each image is treated as a standalone sample.
    - Temporal mode: group multiple phases of the same patient into one sample (requires filenames that encode patient ID and phase).
    """

    def __init__(self, root: str, phase: str, img_size: int = 224,
                 temporal: bool = False, max_T: int = 9,
                 transform=None, phase_sort_key=default_phase_sort_key):
        """
        root/phase/class_name/*.png
        temporal=False: single-frame mode.
        temporal=True: temporal mode; aggregate by patient ID, taking up to `max_T` frames (pad or truncate as needed).
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
            # Temporal mode: aggregate by patient ID.
            # Assume the filename contains patient information; use regex to split patient and phase.
            # Example rule: strip the last underscore suffix, e.g., P001_phase00.png -> patient = P001.
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

            # Trim or pad each (patient, label) sequence to exactly `max_T` frames.
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
            # `self._load_image` returns a tensor normalised to ImageNet stats with shape [3, H, W].
            x = self._load_image(path)           # [3,H,W]
            return x, label, os.path.basename(path)
        else:
            paths, label = self.samples[idx]     # List[str] with length T
            # Each frame produces a [3, H, W] tensor; stacking along dim=0 creates a temporal volume.
            frames = [self._load_image(p) for p in paths]   # List of [3,H,W]
            x = torch.stack(frames, dim=0)       # [T,3,H,W]
            return x, label, [os.path.basename(p) for p in paths]

def build_tms_loader(root: str, phase: str, batch_size: int,
                     img_size: int = 224, temporal: bool = True, max_T: int = 9,
                     num_workers: int = 4, pin_memory: bool = True, shuffle: bool = False):
    dataset = TMSImageFolder(root=root, phase=phase, img_size=img_size,
                             temporal=temporal, max_T=max_T,
                             transform=build_transforms(train=(phase=='train'), img_size=img_size))
    # Loader yields `(x, label, meta)` tuples where:
    #   - temporal=True -> `x` has shape [batch, T, 3, img_size, img_size]
    #   - temporal=False -> `x` has shape [batch, 3, img_size, img_size]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if phase=='train' else False,
                        num_workers=num_workers, pin_memory=pin_memory, drop_last=(phase=='train'))
    return loader, dataset.classes

# ----------------------------------------------------------------------------------------------------
# sequence dataset

import os, re, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

def default_build_transforms(train, img_size):
    ts = []
    if train:
        ts += [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ts += [T.Resize((img_size, img_size))]
    ts += [T.ToTensor()]
    return T.Compose(ts)

_seq_re = re.compile(
    r'^(?P<prefix>BreaDM-(?P<bm>(Ma|Be))-(?P<case>\d+)_VIBRANT)'
    r'(?P<cseq>(\+C(?P<cidx>\d+))?)_p-(?P<pidx>\d+)\.jpg$'
)

def parse_name(fname):
    # Return (BM, case_id, pidx, cidx) where cidx=0 indicates pre-contrast, 1..8 correspond to C1..C8.
    m = _seq_re.match(fname)
    if not m:
        return None
    bm = m.group('bm')      # 'Ma' or 'Be'
    case = m.group('case')  # '1802' ...
    pidx = int(m.group('pidx'))
    cidx = m.group('cidx')
    cidx = int(cidx) if cidx is not None else 0
    return bm, case, pidx, cidx

class TMSSequenceFolder(Dataset):
    def __init__(self, root, phase='train', img_size=224, temporal=True, max_T=9, transform=None):
        super().__init__()
        self.root = root
        self.phase = phase
        self.temporal = temporal
        self.max_T = max_T
        self.img_size = img_size
        self.transform = transform or default_build_transforms(train=(phase=='train'), img_size=img_size)

    # Class subdirectories: assume 'B' is benign and 'M' is malignant.
        self.classes = sorted([d for d in os.listdir(os.path.join(root, phase)) if os.path.isdir(os.path.join(root, phase, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}  # {'B':0,'M':1}

        self.samples = self._build_sequences()

    def _build_sequences(self):
        base = os.path.join(self.root, self.phase)
        items = []
        for cls in self.classes:
            cls_dir = os.path.join(base, cls)
            all_files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
            # Build index: key = (bm, case, pidx), value = {0:pre_path, 1:C1_path, ...}.
            buckets = {}
            for f in all_files:
                parsed = parse_name(f)
                if parsed is None:
                    continue
                bm, case, pidx, cidx = parsed
                key = (bm, case, pidx)
                d = buckets.setdefault(key, {})
                d[cidx] = os.path.join(cls_dir, f)

            for key, frames in buckets.items():
                # Desired frame order: 0..8 (pre + C1..C8).
                order = list(range(min(self.max_T, 9))) if self.temporal else [0]
                paths = []
                ok = True
                for t in order:
                    if t in frames:
                        paths.append(frames[t])
                    else:
                        ok = False
                        break
                if not ok:
                    # To tolerate missing frames, you could fill here with nearest available frames.
                    # By default we skip incomplete sequences to enforce exactly nine frames.
                    continue
                label = self.class_to_idx[cls]
                items.append((paths, label, key))
        return items

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __getitem__(self, idx):
        paths, label, key = self.samples[idx]
        if self.temporal:
            # For temporal consistency consider fixing augmentation randomness across frames.
            # Simplification here: call the same transform independently per frame (minor randomness is typically acceptable).
            frames = [self.transform(self._load_img(p)) for p in paths]  # list of [3,H,W]
            # After stacking, `x` is the temporal clip with shape [T, 3, H, W].
            x = torch.stack(frames, dim=0)  # [T,3,H,W]
        else:
            x = self.transform(self._load_img(paths[0]))  # [3,H,W]
        y = torch.tensor(label, dtype=torch.long)
        meta = {'key': key, 'paths': paths}
        return x, y, meta