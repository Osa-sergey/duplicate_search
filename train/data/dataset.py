import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
from .transform import img_augmentations


def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


path_to_gifs = "../zorin/pics/data/gifs_to_imgs"
path_to_watermarks = "../zorin/pics/data/watermarks"

# Пример использования
all_gifs = get_all_files(path_to_gifs)
all_watermarks = get_all_files(path_to_watermarks)
all_texts = [
    'YOUTUBE',
    'ABOBA',
    'TIKTOK',
    'RUTUBE',
    'READY',
    'SUBSCRIBE',
    'LIKE',
    'LIKE AND SUBSCRIBE',
    'BEST VIDEO EVER'
] 


class KeyframeDataset(Dataset):
    def __init__(self, root_dir, meta_file, is_val = False, width=256, transform=None):
        """
        root_dir: корневая директория с сохраненными ключевыми кадрами
        transform: преобразования для изображений (например, Resize, ToTensor)
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(meta_file)
        self.is_val = is_val
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((width, width)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.width = width

    def __len__(self):
        return self.meta.shape[0]

    def _get_keyframe(self, video_folder):
        """Возвращает случайный ключевой кадр из видео"""
        keyframe_dir = os.path.join(video_folder)
        keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith('.jpg')]
        if len(keyframe_files) == 0:
            print(keyframe_dir)
        keyframe_path = os.path.join(keyframe_dir, random.choice(keyframe_files))
        image = Image.open(keyframe_path)
        return image

    def __getitem__(self, idx):
        uuid = self.meta.loc[idx, 'uuid']
        is_duplicate = self.meta.loc[idx, 'is_duplicate']
        if is_duplicate:
            duplicate_for = self.meta.loc[idx, 'duplicate_for']
        else:
            duplicate_for = None

        anchor_video_folder = f'{self.root_dir}/{uuid}'
        if duplicate_for is not None:
            positive_video_folder = f'{self.root_dir}/{duplicate_for}'
        else:
            positive_video_folder = anchor_video_folder

        anchor_image = self._get_keyframe(anchor_video_folder)
        flag = random.choices([0, 1, 2], weights = [20, 70, 10])

        if flag[0] == 1:
            positive_image = img_augmentations(anchor_image, width=self.width, aug_materials=[all_gifs, all_watermarks, all_texts], pipeline_type='hard')
        elif flag[0] == 2:
            positive_image = self._get_keyframe(positive_video_folder)
        elif flag[0] == 0:
            positive_image = img_augmentations(anchor_image, width=self.width, aug_materials=[all_gifs, all_watermarks, all_texts], pipeline_type='easy')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            if flag[0] == 2:
                positive_image = self.transform(positive_image)
        return anchor_image, positive_image

    

class ValKeyframeDataset(Dataset):
    def __init__(self, root_dir, meta_file,  width=256, transform=None):
        """
        root_dir: корневая директория с сохраненными ключевыми кадрами
        transform: преобразования для изображений (например, Resize, ToTensor)
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(meta_file)
        random.seed(42)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((width, width)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.width = width

    def __len__(self):
        return self.meta.shape[0]

    def _get_keyframe(self, video_folder):
        """Возвращает случайный ключевой кадр из видео"""
        keyframe_dir = os.path.join(video_folder)
        keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith('.jpg')]
        if len(keyframe_files) == 0:
            print(keyframe_dir)
        keyframe_path = os.path.join(keyframe_dir, random.choice(keyframe_files))
        image = Image.open(keyframe_path)
        return image

    def __getitem__(self, idx):
        uuid = self.meta.loc[idx, 'uuid']
        is_duplicate = self.meta.loc[idx, 'is_duplicate']
        if is_duplicate:
            duplicate_for = self.meta.loc[idx, 'duplicate_for']
        else:
            duplicate_for = None

        anchor_video_folder = f'{self.root_dir}/{uuid}'
        if duplicate_for is not None:
            positive_video_folder = f'{self.root_dir}/{duplicate_for}'
        else:
            positive_video_folder = anchor_video_folder
     
        anchor_image = self._get_keyframe(anchor_video_folder)
        flag = random.choices([0, 1, 2], weights = [20, 70, 10])

        if flag[0] == 1:
            positive_image = img_augmentations(anchor_image, width=self.width, aug_materials=[all_gifs, all_watermarks, all_texts], pipeline_type='hard')
        elif flag[0] == 2:
            positive_image = self._get_keyframe(positive_video_folder)
        elif flag[0] == 0:
            positive_image = img_augmentations(anchor_image, width=self.width, aug_materials=[all_gifs, all_watermarks, all_texts], pipeline_type='easy')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            if flag[0] == 2:
                positive_image = self.transform(positive_image)
        return anchor_image, positive_image