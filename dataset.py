import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re
from collections import defaultdict

class BreastDMDataset(Dataset):
    """BreastDM DCE-MRI数据集加载器"""
    
    def __init__(self, 
                 root_dir, 
                 mode='train',
                 img_size=96,  
                 dataset_type='exp_1',  # 'exp_1' (without subtraction), 'exp_2' (with subtraction)
                 augmentation=True):
        """
        Args:
            root_dir: 数据集根目录 (dataset_exp_1 或 dataset_exp_2)
            mode: 'train', 'val', 或 'test'
            img_size: 图像大小
            dataset_type: 数据集类型
            augmentation: 是否使用数据增强
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.augmentation = augmentation and (mode == 'train')
        
        if dataset_type == 'exp_1':
            self.num_sequences = 9  # 1 pre + 8 post
            self.use_subtraction = False
        else:  
            self.num_sequences = 17  # 1 pre + 8 post + 8 subtraction
            self.use_subtraction = True

        self.samples = self._collect_samples()
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.samples)} samples for {mode} set (Dataset: {dataset_type})")
        
    def _collect_samples(self):
        """收集所有样本"""
        samples = []
        data_dir = self.root_dir / self.mode
        if not data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist!")
        for class_name, label in [('B', 0), ('M', 1)]:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue

            cases = defaultdict(lambda: {'pre': [], 'post': defaultdict(list), 'sub': defaultdict(list)})
            
            for img_path in sorted(class_dir.glob("*.jpg")):
                filename = img_path.name
                case_match = re.match(r'(BreaDM-[BM]e-\d+)', filename)
                if not case_match:
                    continue
                case_id = case_match.group(1)
                slice_match = re.search(r'p-(\d+)', filename)
                if not slice_match:
                    continue
                slice_num = int(slice_match.group(1))

                if 'sub' in filename.lower():
                    # Subtraction images (exp_2)
                    contrast_match = re.search(r'C(\d)', filename)
                    if contrast_match:
                        contrast_num = int(contrast_match.group(1))
                        cases[case_id]['sub'][contrast_num].append((slice_num, img_path))
                elif 'VIBRANT+C' in filename:
                    contrast_match = re.search(r'C(\d)', filename)
                    if contrast_match:
                        contrast_num = int(contrast_match.group(1))
                        cases[case_id]['post'][contrast_num].append((slice_num, img_path))
                else:
                    cases[case_id]['pre'].append((slice_num, img_path))

            for case_id, case_data in cases.items():
                all_slices = set()
                for slice_num, _ in case_data['pre']:
                    all_slices.add(slice_num)
 
                for target_slice in sorted(all_slices):
                    sample = {
                        'case_id': case_id,
                        'slice_num': target_slice,
                        'label': label,
                        'pre_contrast': None,
                        'post_contrast': [],
                        'subtraction': []
                    }
                    for slice_num, path in case_data['pre']:
                        if slice_num == target_slice:
                            sample['pre_contrast'] = path
                            break

                    for c_idx in range(1, 9):  # C1 to C8
                        found = False
                        if c_idx in case_data['post']:
                            for slice_num, path in case_data['post'][c_idx]:
                                if slice_num == target_slice:
                                    sample['post_contrast'].append(path)
                                    found = True
                                    break
                        if not found and c_idx <= 4:  
                            break

                    if self.use_subtraction:
                        for c_idx in range(1, 9):
                            if c_idx in case_data['sub']:
                                for slice_num, path in case_data['sub'][c_idx]:
                                    if slice_num == target_slice:
                                        sample['subtraction'].append(path)
                                        break

                    if self.dataset_type == 'exp_1':
                        # exp_1: 1 pre + at least 4 post
                        if sample['pre_contrast'] and len(sample['post_contrast']) >= 4:
                            samples.append(sample)
                    else:  # exp_2
                        # exp_2: full 17 sequence
                        if (sample['pre_contrast'] and 
                            len(sample['post_contrast']) >= 4 and 
                            len(sample['subtraction']) >= 4):
                            samples.append(sample)
        
        return samples
    
    def _get_transforms(self):
        """获取数据变换"""
        if self.augmentation:
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.4
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0)),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.4
                ),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2()
            ])
        
        return transform
    
    def _load_image(self, path):
        """加载图像"""
        img = Image.open(path).convert('L')  # 灰度图
        return np.array(img, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_img = self._load_image(sample['pre_contrast'])

        post_imgs = []
        for i in range(min(8, len(sample['post_contrast']))):
            post_imgs.append(self._load_image(sample['post_contrast'][i]))

        while len(post_imgs) < 8:
            if post_imgs:
                post_imgs.append(post_imgs[-1].copy())
            else:
                post_imgs.append(pre_img.copy())
        
        if self.dataset_type == 'exp_1':
            # Exp-1: 9 seq(1 pre + 8 post)
            all_images = [pre_img] + post_imgs
 
            if self.augmentation:
                seed = np.random.randint(0, 100000)
                transformed_images = []
                for img in all_images:
                    np.random.seed(seed)
                    transformed = self.transform(image=img)['image']
                    transformed_images.append(transformed)
                frames = torch.stack(transformed_images)  # [9, H, W]
            else:
                frames = torch.stack([self.transform(image=img)['image'] for img in all_images])
            
            # [9, 1, H, W] -> [9, H, W]
            frames = frames.squeeze(1) if frames.dim() == 4 else frames
            
            return {
                'frames': frames,
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'case_id': sample['case_id'],
                'slice_num': sample['slice_num']
            }
            
        else:  # exp_2
            # Exp-2: 17 seq (1 pre + 8 post + 8 subtraction)
            sub_imgs = []
            for post_img in post_imgs:
                sub_img = post_img - pre_img
                sub_imgs.append(sub_img)
            
            all_images = [pre_img] + post_imgs + sub_imgs

            if self.augmentation:
                seed = np.random.randint(0, 100000)
                transformed_images = []
                for img in all_images:
                    np.random.seed(seed)
                    transformed = self.transform(image=img)['image']
                    transformed_images.append(transformed)
                frames = torch.stack(transformed_images)  # [17, H, W]
            else:
                frames = torch.stack([self.transform(image=img)['image'] for img in all_images])
            
            frames = frames.squeeze(1) if frames.dim() == 4 else frames
            main_frames = frames[:9]  
            subtraction = frames[9:].mean(dim=0, keepdim=True)  
            
            return {
                'frames': main_frames,
                'subtraction': subtraction,
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'case_id': sample['case_id'],
                'slice_num': sample['slice_num']
            }