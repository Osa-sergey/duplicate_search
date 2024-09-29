import pandas as pd
import os
import os
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import pandas as pd
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import augly
from torch.utils.data import Dataset

class KeyframeValidationDataset(Dataset):
    def __init__(self, root_dir, meta_file, width=256, transform=None):
        """
        root_dir: корневая директория с сохраненными ключевыми кадрами
        transform: преобразования для изображений (например, Resize, ToTensor)
        """
        self.root_dir = root_dir
        self.meta = pd.read_csv(meta_file)
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

    def _get_keyframes(self, video_folder):
        """Возвращает все ключевые кадр из видео"""
        keyframe_dir = os.path.join(video_folder)
        keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith('.jpg')]
        if len(keyframe_files) == 0:
            print(keyframe_dir)
        keyframe_paths = [os.path.join(keyframe_dir, keyframe_file) for keyframe_file in keyframe_files]
        images = [Image.open(keyframe_path) for keyframe_path in keyframe_paths]
        return images

    def __getitem__(self, idx):
        uuid = self.meta.loc[idx, 'uuid']
        if isinstance(uuid, pd.Series):
            uuid = uuid.iloc[0]
        is_duplicate = self.meta.loc[idx, 'is_duplicate']
        if isinstance(is_duplicate, pd.Series):
            is_duplicate = is_duplicate.iloc[0]
        # print(f"{is_duplicate = }")
        # print(f"{self.meta.loc[idx]}")
        if is_duplicate:
            duplicate_for = self.meta.loc[idx, 'duplicate_for']
            if isinstance(duplicate_for, pd.Series):
                duplicate_for = duplicate_for.iloc[0]
        else:
            duplicate_for = None
        # print(f"{duplicate_for = }")
    
        anchor_video_folder = f'{self.root_dir}/{uuid}'
        anchor_images = self._get_keyframes(anchor_video_folder)

        if self.transform:
            anchor_images = [self.transform(anchor_image) for anchor_image in anchor_images]
        
        return {"input_ids" : anchor_images, 
                "uuid" : uuid, 
                "timestamp" : self.meta.loc[idx, "created"], 
                "gt" : duplicate_for}
