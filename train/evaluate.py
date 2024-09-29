# TODO
# FAISS start

import numpy as np
import uuid
from datetime import datetime

def find_closest_vector(index, query_vector):
    distances, indices = index.search(query_vector.reshape(1, -1), k=1)
    return distances, indices

# FAISS end

import numpy as np
import torch
from tqdm import tqdm

from torchmetrics.functional import pairwise_cosine_similarity

# процесс валидации
# 80% данных train, 20% val (выбор случайно)
# train f1 - метрика на дубликатах внутри (train, train)
# val f1 - метрика на дубликатах на парах (train, val) и (val, val)

# def my_collate_fn(elems):
#     batch = {}
#     input_ids = [elem["input_ids"] for elem in elems]
#     batch["input_ids"] = torch.stack(input_ids, dim=0) # yes...
#     batch["gt"] = [elem["gt"] for elem in elems]
#     batch["timestamp"] = [elem["timestamp"] for elem in elems]
#     batch["uuid"] = [elem["uuid"] for elem in elems]
#     return batch

def one_pic_collate_fn(elems):
    assert len(elems) == 1
    batch = {}
    batch["input_ids"] = torch.stack(elems[0]["input_ids"])
    batch["gt"] = [elems[0]["gt"]]
    batch["uuid"] = [elems[0]["uuid"]]
    batch["timestamp"] = [elems[0]["timestamp"]]
    return batch

def calc_f1_score(model, gts, timestamps, uuids, pairwise_sim, nearests, threshold):
    pred, gt = [], []
    
    for nearest_anchor, nearest_pred in tqdm(enumerate(nearests)):
        gt.append(gts[nearest_anchor] is not None)
        sim = pairwise_sim[nearest_anchor, nearest_pred]
        # print(sim)
        if sim < threshold:
            pred.append(0)
        elif gts[nearest_anchor] == uuids[nearest_pred.item()]:
            pred.append(1)
        else:
            pred.append(1 - gt[-1])
    
    print(f"{sum(gt) / len(gt) = }")
    print(f"{sum(pred) / len(pred) = }")
    from sklearn.metrics import recall_score as recall
    from sklearn.metrics import precision_score as precision
    from sklearn.metrics import f1_score
    pred = np.array(pred)
    gt = np.array(gt)
    metrics = {
            "f1_score" : f1_score(gt, pred),
            "precision" : precision(gt, pred),
            "recall" : recall(gt, pred),
            "tp" : sum(gt * pred),
            "fn" : sum(gt * (1 - pred)),
            "tn" : sum((1 - gt) * (1 - pred)),
            "fp" : sum((1 - gt) * pred),
    }
    print(f"{threshold = }, {metrics = }")
    return metrics

        
def evaluate(model, scoring_dataset):
    scoring_dataloader = DataLoader(scoring_dataset, collate_fn=one_pic_collate_fn, batch_size=1, num_workers=16)
    all_embeds = torch.zeros((len(scoring_dataset), 512)).cuda()
    gts = []
    timestamps = []
    uuids = []
    for num_batch, batch in tqdm(enumerate(scoring_dataloader)):
        embeds = model.forward(batch["input_ids"].cuda())
        all_embeds[num_batch: num_batch + 1] = torch.mean(embeds, dim=0)
        # all_embeds[num_batch: num_batch + 1] = torch.nn.functional.normalize(all_embeds)
        gts.extend(batch["gt"])
        timestamps.extend(batch["timestamp"])
        uuids.extend(batch["uuid"])
        
    # SORT BASED ON TIMESTMAP
    
    print(f"before pairwise calc, {all_embeds.shape = }")
    pairwise_sim = pairwise_cosine_similarity(all_embeds)
    print("after pairwise calc")
    nearest_mask = torch.tril(
        torch.ones(len(scoring_dataset), len(scoring_dataset))
        # torch.ones(batch_size, batch_size) # DEBUG 3
    ).cuda()
    print("mask created")
    pairwise_sim *= nearest_mask
    print("sim masked")
    pairwise_sim = pairwise_sim.cpu()
    print("to cpu")
    nearests = torch.argmax(pairwise_sim, dim = 1).cpu()
    print("nearests found")
    threshold_grid = np.linspace(-0.6, 0.9, 21)
    
    # print(pairwise_sim)
    # raise OSError("It's time to stop!")
    
    scores = []
    for threshold in threshold_grid:
        f1_score = calc_f1_score(model, gts, timestamps, uuids, pairwise_sim, nearests, threshold)
        scores.append(f1_score)
        
    for threshold, score in zip(threshold_grid, scores):
        print(f"{threshold = }; {score = }")
    
    return scores

# from aug import img_augmentations, pad, InvertColors, CropAndPad, BaseTransform, RandomCompose
from keyframedataset import KeyframeValidationDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader

# Модель на основе EfficientNet для получения эмбеддингов
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


val_dataset = KeyframeValidationDataset('../zorin/train', '../zorin/val_split.csv', width=256)
# states = torch.load('checkpoints_final/vit/model_71.pth')
# state_dict = {k.replace('module.', ''): v for k, v in states.items()}
from models import load_swinv2, load_vit
# vit_checkpoint = 'checkpoints_pretrain/vit_v68.torchscript.pt' 
# model = load_vit(vit_checkpoint)
# model.load_state_dict(state_dict)


# from models import EfficientNetEmbeddingXL
# model = EfficientNetEmbeddingXL()
# state = torch.load('checkpoints_final/efficientnetxl/model_45.pth')
# model.load_state_dict(state)
# model.load_state_dict(state_dict)

# states = torch.load('../zorin/epoch_36.pth')['state_dict']
# state_dict = {k.replace('module.', ''): v for k, v in states.items()}
# model = EfficientNetEmbedding()
# model.load_state_dict(state_dict)

# from models import load_swinv2
# model = load_swinv2(swinv2_checkpoint)
# states = torch.load('../zorin/epoch_36.pth')['state_dict']
# state_dict = {k.replace('module.', ''): v for k, v in states.items()}
# model = EfficientNetEmbedding()
# model.load_state_dict(state_dict)

# import sys
# sys.path.insert(0, '../rusakov')
# # from models import load_vit
# # vit_checkpoint = '../rusakov/model_checkpoints/vit_v68.torchscript.pt' 
# # model = load_vit(vit_checkpoint)
# # state_dict = torch.load('../rusakov/checkpoints/vit/debug/epoch_38.pth')['state_dict']
# # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# # model.load_state_dict(state_dict)

# from models import load_swinv2
swinv2_checkpoint = 'checkpoints_pretrain/swinv2_v115.torchscript.pt'
model = load_swinv2(swinv2_checkpoint)
# state_dict = torch.load('../rusakov/checkpoints/swin/debug/epoch_10.pth')['state_dict']
# state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(state_dict)

model.eval()
model.to("cuda")

print(f"{len(val_dataset) = }")

with torch.no_grad():
    evaluate(model, val_dataset)