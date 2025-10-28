import os, re, random
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

def default_build_transforms(train, img_size):
    """
    仅做确定性的 Resize + ToTensor。
    随机增强（几何/颜色/噪声）在 __getitem__ 中通过 TemporalAugment 统一作用于整段序列。
    """
    ts = [T.Resize((img_size, img_size)), T.ToTensor()]
    return T.Compose(ts)

# ---------------- 文件名解析 ----------------
_re_vibrant_pre = re.compile(
    r'^(?P<prefix>BreaDM-(?P<bm>(Ma|Be))-(?P<case>\d+)_VIBRANT)_p-(?P<pidx>\d+)\.jpg$'
)
_re_vibrant_ck = re.compile(
    r'^(?P<prefix>BreaDM-(?P<bm>(Ma|Be))-(?P<case>\d+)_VIBRANT)\+C(?P<cidx>\d+)_p-(?P<pidx>\d+)\.jpg$'
)
_re_subk = re.compile(
    r'^(?P<prefix>BreaDM-(?P<bm>(Ma|Be))-(?P<case>\d+)_SUB)(?P<sidx>\d+)_p-(?P<pidx>\d+)\.jpg$'
)

def parse_name_any(fname):
    m = _re_vibrant_pre.match(fname)
    if m:
        return {'bm': m.group('bm'), 'case': m.group('case'),
                'pidx': int(m.group('pidx')), 'type': 'pre', 'idx': 0}
    m = _re_vibrant_ck.match(fname)
    if m:
        return {'bm': m.group('bm'), 'case': m.group('case'),
                'pidx': int(m.group('pidx')), 'type': 'vibrant', 'idx': int(m.group('cidx'))}
    m = _re_subk.match(fname)
    if m:
        return {'bm': m.group('bm'), 'case': m.group('case'),
                'pidx': int(m.group('pidx')), 'type': 'sub', 'idx': int(m.group('sidx'))}
    return None

# ---------------- 时序一致增强模块 ----------------
class TemporalAugment:
    """
    对 [T,3,H,W] 的序列做“同一随机参数”的增强，确保时序一致。
    """
    def __init__(
        self,
        do_hflip=True, p_hflip=0.5,
        do_affine=True, degrees=12.0, translate=0.06, scale=(0.92, 1.08), shear=5.0, p_affine=0.7,
        do_crop=True, crop_ratio=(0.85, 1.0), p_crop=0.8,
        do_bcj=True, brightness=0.12, contrast=0.12, p_bcj=0.5,
        do_gaussian=True, gauss_std=0.01, p_gaussian=0.3,
        do_blur=False, blur_radius=0.6, p_blur=0.2,
        img_size=224
    ):
        self.do_hflip = do_hflip
        self.p_hflip = float(p_hflip)

        self.do_affine = do_affine
        self.degrees = float(degrees)
        self.translate = float(translate)  # 相对 H/W 比例
        self.scale = tuple(scale)
        self.shear = float(shear)
        self.p_affine = float(p_affine)

        self.do_crop = do_crop
        self.crop_ratio = crop_ratio
        self.p_crop = float(p_crop)

        self.do_bcj = do_bcj
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p_bcj = float(p_bcj)

        self.do_gaussian = do_gaussian
        self.gauss_std = float(gauss_std)
        self.p_gaussian = float(p_gaussian)

        self.do_blur = do_blur
        self.blur_radius = float(blur_radius)
        self.p_blur = float(p_blur)

        self.img_size = int(img_size)

    def _rand_affine_params(self, H, W):
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = self.translate * W
        max_dy = self.translate * H
        trans = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)
        return angle, trans, scale, shear

    def _center_crop_params(self, H, W):
        r = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        new_h = max(1, int(H * r))
        new_w = max(1, int(W * r))
        top = (H - new_h) // 2
        left = (W - new_w) // 2
        return top, left, new_h, new_w

    def _apply_pil_blur(self, x):
        # x: [T,3,H,W], 0..1
        Tn, C, H, W = x.shape
        out = []
        for t in range(Tn):
            img = TF.to_pil_image(x[t].clamp(0,1))
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            out.append(TF.to_tensor(img))
        return torch.stack(out, dim=0)

    def __call__(self, x):
        """
        x: torch.Tensor [T,3,H,W], 值域[0,1]
        返回：增强后的同型张量
        """
        assert torch.is_tensor(x) and x.dim()==4 and x.size(1)==3, 'TemporalAugment expects [T,3,H,W]'
        Tn, C, H, W = x.shape

        # 一致水平翻转
        if self.do_hflip and random.random() < self.p_hflip:
            x = torch.flip(x, dims=[-1])

        # 一致随机仿射（同参）
        if self.do_affine and random.random() < self.p_affine:
            angle, trans, scale, shear = self._rand_affine_params(H, W)
            x = torch.stack([TF.affine(x[t], angle=angle, translate=(int(trans[0]), int(trans[1])),
                                       scale=scale, shear=shear, interpolation=T.InterpolationMode.BILINEAR)
                             for t in range(Tn)], dim=0)

        # 一致中心随机裁剪后再Resize回原尺寸
        if self.do_crop and random.random() < self.p_crop:
            top, left, new_h, new_w = self._center_crop_params(H, W)
            x = x[:, :, top:top+new_h, left:left+new_w]
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # 亮度/对比度轻抖动（同参）
        if self.do_bcj and random.random() < self.p_bcj:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            x = (c * x + (b - 1.0)).clamp(0.0, 1.0)

        # 轻噪声
        if self.do_gaussian and random.random() < self.p_gaussian:
            noise = torch.randn_like(x) * self.gauss_std
            x = (x + noise).clamp(0.0, 1.0)

        # 轻模糊（可选）
        if self.do_blur and random.random() < self.p_blur:
            x = self._apply_pil_blur(x).to(x.device)

        return x

class TemporalConsistentRandomErasing:
    """
    对整段序列在同一空间位置做一次擦除（同参、同位置）。
    """
    def __init__(self, p=0.25, area=(0.02, 0.05), aspect=(0.5, 2.0), value=0.0):
        self.p = float(p)
        self.area = area
        self.aspect = aspect
        self.value = float(value)

    def __call__(self, x):  # x: [T,3,H,W], 0..1
        if random.random() >= self.p:
            return x
        Tn, C, H, W = x.shape
        area_img = H * W
        for _ in range(10):
            target_area = random.uniform(self.area[0], self.area[1]) * area_img
            aspect = random.uniform(self.aspect[0], self.aspect[1])
            h = int(round((target_area * aspect) ** 0.5))
            w = int(round((target_area / aspect) ** 0.5))
            if h < H and w < W and h > 0 and w > 0:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                x[:, :, top:top+h, left:left+w] = self.value
                return x
        return x

# ---------------- 数据集 ----------------
class TMSSequenceFolder(Dataset):
    def __init__(self, root, phase='train', img_size=224, temporal=True, max_T=9,
                 transform=None, seq_source='exp1_vibrant',
                 enable_aug=True,
                 hard_keys=None, hard_strong_aug=False):
        """
        seq_source:
          - 'exp1_vibrant'  : [pre, C1..C8] (T=9)
          - 'exp2_sub'      : [SUB1..SUB8]  (T=8)
          - 'exp2_pre_sub'  : [pre, SUB1..SUB8] (T=9)
          - 'exp2_17'       : [pre, C1..C8, SUB1..SUB8] (T=17)
        """
        super().__init__()
        self.root = root
        self.phase = phase
        self.temporal = temporal
        self.max_T = max_T
        self.seq_source = seq_source
        self.img_size = img_size
        self.transform = transform or default_build_transforms(train=(phase=='train'), img_size=img_size)

        # 时序一致增强器（仅训练用，可通过 enable_aug 控制）
        self.enable_aug = bool(enable_aug and phase == 'train')
        self.temporal_aug = TemporalAugment(
            do_hflip=True, p_hflip=0.5,
            do_affine=True, degrees=12.0, translate=0.06, scale=(0.92, 1.08), shear=5.0, p_affine=0.7,
            do_crop=True, crop_ratio=(0.85, 1.0), p_crop=0.8,
            do_bcj=True, brightness=0.12, contrast=0.12, p_bcj=0.5,
            do_gaussian=True, gauss_std=0.01, p_gaussian=0.3,
            do_blur=False, blur_radius=0.6, p_blur=0.2,
            img_size=img_size
        )
        # 针对难例的更强增强（可选）
        self.hard_keys = set(hard_keys) if hard_keys is not None else set()
        self.hard_strong_aug = bool(hard_strong_aug)
        self.temporal_aug_hard = TemporalAugment(
            do_hflip=True, p_hflip=0.6,
            do_affine=True, degrees=18.0, translate=0.08, scale=(0.9, 1.12), shear=8.0, p_affine=0.8,
            do_crop=True, crop_ratio=(0.82, 1.0), p_crop=0.9,
            do_bcj=True, brightness=0.18, contrast=0.18, p_bcj=0.6,
            do_gaussian=True, gauss_std=0.015, p_gaussian=0.4,
            do_blur=False, blur_radius=0.8, p_blur=0.25,
            img_size=img_size
        )
        self.t_erase = TemporalConsistentRandomErasing(p=0.25, area=(0.02, 0.05), aspect=(0.5, 2.0), value=0.0)

        base = os.path.join(self.root, self.phase)
        self.classes = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}  # {'B':0,'M':1}
        self.samples = self._build_sequences()

    def _build_sequences(self):
        base = os.path.join(self.root, self.phase)
        items = []
        for cls in self.classes:
            cls_dir = os.path.join(base, cls)
            all_files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
            buckets = {}
            for f in all_files:
                info = parse_name_any(f)
                if info is None:
                    continue
                key = (info['bm'], info['case'], info['pidx'])
                d = buckets.setdefault(key, {'pre': None, 'vibrant': {}, 'sub': {}})
                if info['type'] == 'pre':
                    d['pre'] = os.path.join(cls_dir, f)
                elif info['type'] == 'vibrant':
                    d['vibrant'][info['idx']] = os.path.join(cls_dir, f)
                elif info['type'] == 'sub':
                    d['sub'][info['idx']] = os.path.join(cls_dir, f)

            for key, frames in buckets.items():
                paths = None
                if self.seq_source == 'exp1_vibrant':
                    need = min(self.max_T, 9)
                    if not frames['pre']:
                        continue
                    seq = [frames['pre']]
                    ok = True
                    for k in range(1, 9):
                        if k in frames['vibrant']:
                            seq.append(frames['vibrant'][k])
                        else:
                            ok = False; break
                    if not ok:
                        continue
                    paths = seq[:need]

                elif self.seq_source == 'exp2_sub':
                    order = list(range(1, 9))
                    need = min(self.max_T, len(order))
                    ok = True
                    seq = []
                    for k in order:
                        if k in frames['sub']:
                            seq.append(frames['sub'][k])
                        else:
                            ok = False; break
                    if not ok:
                        continue
                    paths = seq[:need]

                elif self.seq_source == 'exp2_pre_sub':
                    need = min(self.max_T, 9)
                    if not frames['pre']:
                        continue
                    ok = True
                    seq = [frames['pre']]
                    for k in range(1, 9):
                        if k in frames['sub']:
                            seq.append(frames['sub'][k])
                        else:
                            ok = False; break
                    if not ok:
                        continue
                    paths = seq[:need]

                elif self.seq_source == 'exp2_17':
                    need = min(self.max_T, 17)
                    if not frames['pre']:
                        continue
                    ok = True
                    seq = [frames['pre']]
                    for k in range(1, 9):
                        if k in frames['vibrant']:
                            seq.append(frames['vibrant'][k])
                        else:
                            ok = False; break
                    if ok:
                        for k in range(1, 9):
                            if k in frames['sub']:
                                seq.append(frames['sub'][k])
                            else:
                                ok = False; break
                    if not ok:
                        continue
                    paths = seq[:need]
                else:
                    raise ValueError(f'Unknown seq_source: {self.seq_source}')

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
            imgs = [self._load_img(p) for p in paths]
            frames = [self.transform(im) for im in imgs]  # [T]*[3,H,W]
            x = torch.stack(frames, dim=0)  # [T,3,H,W]
            # 时序一致随机增强（仅训练阶段）
            if self.enable_aug:
                # 难例使用更强增强（可选）
                if self.hard_strong_aug and key in self.hard_keys:
                    x = self.temporal_aug_hard(x)
                else:
                    x = self.temporal_aug(x)
                x = self.t_erase(x)
        else:
            x = self.transform(self._load_img(paths[0]))  # [3,H,W]

        y = torch.tensor(label, dtype=torch.long)
        meta = {'key': key, 'paths': paths}
        return x, y, meta

# 兼容旧接口名
build_transforms = default_build_transforms