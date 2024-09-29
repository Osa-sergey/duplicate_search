import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import augly
import augly.image
from data.dataset import KeyframeDataset, ValKeyframeDataset
from models import EfficientNetEmbedding, EfficientNetEmbeddingXL
from models import load_swinv2, load_vit


def contrast_loss_fn(emb_a, emb_b, temperature, m=None):
    bz = emb_a.size(0)
    emb = torch.cat([emb_a, emb_b], dim=0)  # 2xbz
    sims = emb @ emb.t()
    diag = torch.eye(sims.size(0)).to(sims.device)

    small_value = torch.tensor(-10000.).to(sims.device).to(sims.dtype)
    sims = torch.where(diag.eq(0), sims, small_value)
    gt = torch.cat([torch.arange(bz) + bz, torch.arange(bz)], dim=0).to(sims.device)
    # mask = torch.cat([mask, mask], dim=0).bool()

    # if args.margin > 0:
    #     loss_ = F.cross_entropy((sims - diag * args.margin) / temperature, gt, reduction="none").mean()
    # else:
    loss_ = F.cross_entropy(sims / temperature, gt, reduction="none").mean()

    return loss_


def entropy_loss_fn(sims):
    device = sims.device
    diag = torch.eye(sims.size(0)).to(device)
    # local_mask = mask[:, None] * mask[None, :] * (1 - diag)  # 加上这行 matching会提高，descriptor降低
    local_mask = (1 - diag)
    small_value = torch.tensor(-10000.).to(device).to(sims.dtype)
    max_non_match_sim = torch.where(local_mask.bool(), sims, small_value).max(dim=1)[0]
    closest_distance = (1 / 2 - max_non_match_sim / 2).clamp(min=1e-6).sqrt()
    entropy_loss_ = -closest_distance.log().mean() * 30
    return entropy_loss_


def train_step(model, image_a, image_b):
    bz = image_a.size(0)

    cat_x = torch.cat([image_a, image_b], dim=0)

    embeds = model(x=cat_x)
    embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)
    emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
    emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2*bz]

    sims_norm = emb_a_norm @ emb_b_norm.t()

    entropy_loss_ = entropy_loss_fn(sims_norm)

    ici_loss_ = contrast_loss_fn(emb_a_norm, emb_b_norm, 0.05) * 1

    return ici_loss_ + entropy_loss_


import argparse
def main():
    parser = argparse.ArgumentParser(description='Train an embedding network')
    parser.add_argument('--model_type', type=str, default='efficientnet', help='Type of the model to use') # efficientnet, #swin #vit
    parser.add_argument('--batch_size', type=int, default=128, help='Type of the model to use')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    model_type = args.model_type
    if model_type == 'efficientnet':
        model = EfficientNetEmbedding(embedding_dim=128).to(device)
    elif model_type == 'swin':
        model = load_swinv2('checkpoints_pretrain/swinv2_v115.torchscript.pt').to(device)
    elif model_type == 'vit':
        model = load_vit('checkpoints_pretrain/vit_v68.torchscript.pt').to(device)
    elif model_type == 'efficientnetxl':
        model = EfficientNetEmbeddingXL(embedding_dim=128).to(device)
    else:
        raise ValueError('not recognize model type')
    if model_type != 'vit':
        train_dataset = KeyframeDataset('../zorin/train', '../zorin/train_split.csv', width=256)
        val_dataset = ValKeyframeDataset('../zorin/train', '../zorin/val_split.csv', width=256)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    else:
        train_dataset = KeyframeDataset('../zorin/train', '../zorin/train_split.csv', width=384)
        val_dataset = ValKeyframeDataset('../zorin/train', '../zorin/val_split.csv', width=384)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        running_loss = 0.0
        num_steps = 0
        for i, (anchor_img, positive_img) in enumerate(train_dataloader):
            anchor_img, positive_img = anchor_img.to(device), positive_img.to(device)
            num_steps += 1

            loss = train_step(model, anchor_img, positive_img)
            cpu_loss = loss.cpu().item()
            # Обновляем веса
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Train step: {num_steps}/{len(train_dataloader)} loss: {cpu_loss}')
            running_loss += loss.item()
        with torch.no_grad():
            val_loss = 0
            val_num_steps = 0
            for i, (val_anchor_img, val_positive_img) in enumerate(val_dataloader):
                val_num_steps += 1
                val_anchor_img, val_positive_img = val_anchor_img.to(device), val_positive_img.to(device)
                val_loss += train_step(model, val_anchor_img, val_positive_img).cpu().item()
            # progress_bar.set_description(f'Val step: {num_steps}/{len(train_dataloader)} loss: {cpu_loss}')
            print(f'Val step: {val_num_steps/len(val_dataloader)}, train_loss: {cpu_loss}, val_loss: {val_loss/len(val_dataloader)}')
      
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader)}')
        os.makedirs(f'checkpoints_final/{model_type}', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints_final/{model_type}/model_{epoch}.pth')
    print('Обучение завершено.')
        

if __name__ == '__main__':
    main()