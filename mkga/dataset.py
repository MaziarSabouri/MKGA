import os
import json
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mode='train', img_size=512):
    if mode == 'train':
        return A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            ], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class ThyroidDataset(Dataset):
    def __init__(self, root_dir, mode='train', split='all', val_ratio=0.2, fraction=1.0,
                 transform=None, data_source='ThyroidXL', use_external_masks=False, binary_tirads=False):
        self.root_dir = root_dir
        self.mode = mode
        self.split = split
        self.val_ratio = val_ratio
        self.fraction = fraction
        self.img_size = 512
        self.transform = get_transforms(mode=mode, img_size=self.img_size)
        self.data_source = data_source
        self.use_external_masks = use_external_masks
        self.binary_tirads = binary_tirads
        self.samples = []
        self.ignore_index = -1
        self.position_map = {"right lobe": 0, "left lobe": 1, "isthmus": 2}

        sources = ['ThyroidXL', 'DDTI'] if self.data_source == 'all' else [self.data_source]
        for source in sources:
            self._load_source(source)

        print(f"Dataset: {self.data_source} | Mode: {mode} | Split: {split} | "
              f"Fraction: {self.fraction} | Samples: {len(self.samples)}")

    def _get_position_label(self, pos_str):
        if not isinstance(pos_str, str):
            return self.ignore_index
        return self.position_map.get(pos_str.lower().strip(), self.ignore_index)

    def _load_source(self, dataset_type):
        if dataset_type == 'ThyroidXL':
            base = os.path.join(self.root_dir, 'Dataset/ThyroidXL')
            json_path = os.path.join(base, 'stats', 'id2info_eng_clean.json')
            img_dir = os.path.join(base, self.mode, 'images')
            mask_dir = os.path.join(base, self.mode, 'masks')
            has_masks = True
        elif dataset_type == 'DDTI':
            base = os.path.join(self.root_dir, 'Dataset/DDTI')
            json_path = os.path.join(base, 'stats', 'ddti_dataset.json')
            sub_folder = 'test' if self.mode == 'test' else 'train'
            if self.mode == 'train' and not os.path.exists(os.path.join(base, 'train')):
                if self.split != 'all':
                    return
            img_dir = os.path.join(base, sub_folder, 'images')
            mask_dir = os.path.join(base, sub_folder, 'masks')
            has_masks = True
        else:
            return

        if not os.path.exists(json_path):
            return
        with open(json_path, 'r') as f:
            data = json.load(f)

        if os.path.exists(img_dir):
            all_files = sorted(os.listdir(img_dir))
            if self.fraction < 1.0:
                random.Random(42).shuffle(all_files)
                num_keep = max(int(len(all_files) * self.fraction), 5)
                all_files = all_files[:num_keep]
            if self.mode == 'train' and self.split in ['train', 'val']:
                random.Random(42).shuffle(all_files)
                split_idx = int(len(all_files) * (1 - self.val_ratio))
                if self.split == 'train':
                    valid_files = set(all_files[:split_idx])
                elif self.split == 'val':
                    valid_files = set(all_files[split_idx:])
            else:
                valid_files = set(all_files)
        else:
            valid_files = set()

        for patient_id, info in data.items():
            age = info.get('age')
            gender = info.get('gender')
            nodule = info.get('nodule_1')
            pos_label = self.ignore_index
            tirads = 0
            if nodule:
                pos_label = self._get_position_label(nodule.get('Position'))
                try:
                    tirads = int(nodule.get('TIRADS')) if nodule.get('TIRADS') is not None else 0
                except ValueError:
                    tirads = 0

            if self.binary_tirads:
                if tirads == 0:
                    tirads = -1
                elif tirads <= 3:
                    tirads = 0
                elif tirads >= 4:
                    tirads = 1
                else:
                    tirads = -1

            images = info.get('images', [])
            for img_name in images:
                if img_name in valid_files:
                    mask_path = None
                    if has_masks and mask_dir and os.path.exists(mask_dir):
                        c1 = os.path.join(mask_dir, img_name)
                        c2 = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")
                        if os.path.exists(c2):
                            mask_path = c2
                        elif os.path.exists(c1):
                            mask_path = c1

                    self.samples.append({
                        'img_path': os.path.join(img_dir, img_name),
                        'mask_path': mask_path,
                        'has_mask': (mask_path is not None),
                        'age': age, 'gender': gender,
                        'position': pos_label, 'tirads': tirads,
                        'dataset': dataset_type
                    })

    def _pad_to_square(self, img, is_mask=False):
        h, w = img.shape[:2]
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        val = 0 if is_mask else [0, 0, 0]
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=val)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._pad_to_square(img, False)

        if item['has_mask'] and item['mask_path']:
            mask = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = self._pad_to_square(mask, True)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if item['dataset'] == 'DDTI' and self.mode == 'train' and not self.use_external_masks:
            mask_valid = False
        else:
            mask_valid = item['has_mask']

        age_val = item['age'] / 100.0 if item['age'] is not None else -1.0
        gen_val = item['gender'] if item['gender'] is not None else -1.0

        return {
            'image': img,
            'mask': mask,
            'mask_valid': torch.tensor(mask_valid, dtype=torch.bool),
            'age': torch.tensor(age_val, dtype=torch.float32),
            'gender': torch.tensor(gen_val, dtype=torch.float32),
            'position': torch.tensor(item['position'], dtype=torch.long),
            'tirads': torch.tensor(item['tirads'], dtype=torch.long)
        }
