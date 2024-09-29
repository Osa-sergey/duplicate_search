import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class EfficientNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EfficientNetEmbedding, self).__init__()
        # Загружаем предобученную EfficientNet
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Убираем последний слой, чтобы получать эмбеддинги
        self.efficientnet.classifier = nn.Identity()
        # Линейный слой для уменьшения размерности до embedding_dim
        self.fc = nn.Linear(1280, embedding_dim)
    
    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Нормализация эмбеддингов
    
class EfficientNetEmbeddingXL(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EfficientNetEmbeddingXL, self).__init__()
        # Загружаем предобученную EfficientNet
        self.efficientnet = models.efficientnet_v2_l(pretrained=True, weights='EfficientNet_V2_L_Weights.DEFAULT')
        # Убираем последний слой, чтобы получать эмбеддинги
        self.efficientnet.classifier = nn.Identity()
        # Линейный слой для уменьшения размерности до embedding_dim
        self.fc = nn.Linear(1280, embedding_dim)
    
    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Нормализация эмбеддингов