import logging
import sys, os
import argparse
import random

import torch
# from vsc.baseline.model_factory.utils import build_model, build_dataset
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv import Config
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from modeling import EMA
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--t', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--instance_mask', action='store_true', default=False)
    parser.add_argument('--entropy_loss', action='store_true', default=False)
    parser.add_argument('--do_ema', action='store_true', default=False)
    parser.add_argument('--do_fgm', action='store_true', default=False)
    parser.add_argument('--fmg_scale', type=float, default=0.5)
    parser.add_argument('--ema_scale', type=float, default=0.99)
    parser.add_argument('--positive_weight', type=float, default=1.)
    parser.add_argument('--entropy_weight', type=float, default=30)
    parser.add_argument('--ici_weight', type=float, default=1.)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--concat_dataset', action='store_true', default=False)
    parser.add_argument('--product_loss', action='store_true', default=False)

    args = parser.parse_args()
    return args


args = parse_args()

work_dir = args.work_dir
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
print_freq = args.print_freq
resume = args.resume if args.resume != '' else None
warmup_ratio = args.warmup_ratio


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


def all_gather(local_rank, world_size, **tensors):
    tensors = list(tensors.values())
    _dims = [t.shape[-1] for t in tensors]
    tensors = torch.cat(tensors, dim=-1)
    tensors_all = [torch.zeros_like(tensors) for _ in range(world_size)]
    dist.all_gather(tensors_all, tensors)
    tensors_all[local_rank] = tensors
    tensors_all = torch.cat(tensors_all, dim=0)

    results = list()
    dimStart = 0
    assert sum(_dims) == tensors_all.shape[-1]
    for d in _dims:
        results.append(tensors_all[..., dimStart: dimStart + d])
        dimStart += d

    return tuple(results)


world_size = int(os.environ['WORLD_SIZE'])
args.rank = int(os.environ['RANK'])
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl')


cfg = Config.fromfile(args.config)
cfg.local_rank = args.local_rank

# assert not os.path.exists(f'checkpoints/{cfg.experiment_name}/{cfg.run_name}'), 'change experiment name or remove old checkpoints'
if args.rank == 0:
    os.system("mkdir -p %s" % work_dir)
    os.system(f"mkdir -p %s/checkpoints/{cfg.experiment_name}/{cfg.run_name}" % work_dir)
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    os.system(f"mkdir -p %s/logs/{cfg.experiment_name}/{cfg.run_name}" % work_dir)
    fh = logging.FileHandler(f'{work_dir}/logs/{cfg.experiment_name}/{cfg.run_name}/log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

train_dataset = cfg.train_dataset
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=train_sampler)
val_dataset = cfg.val_dataset
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=val_sampler)


model = cfg.model
model.cuda(args.local_rank)
ema = EMA(model, args.ema_scale)
ema.register()


opt = AdamW(model.parameters(), lr=lr)
batch_size = batch_size * world_size
stepsize = (len(train_dataset) // batch_size + 1)
total_steps = (len(train_dataset) // batch_size + 1) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio * total_steps,
                                            num_training_steps=total_steps)

model = DDP(model, find_unused_parameters=True)
if args.checkpointing:
    model._set_static_graph()
scaler = torch.cuda.amp.GradScaler()
start_epoch = 0
ckpt = None

if resume:
    ckpt = torch.load(resume, map_location='cpu')
elif os.path.exists(work_dir + '/last.txt'):
    f = open(work_dir + '/last.txt')
    e = int(f.readline())
    f.close()
    ckpt = torch.load(work_dir + '/checkpoints/epoch_%d.pth' % e, map_location='cpu')
if ckpt is not None:
    model.load_state_dict(ckpt['state_dict'])
    opt.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    del ckpt


def contrast_loss_fn(emb_a, emb_b, temperature, mask, weights=None):
    bz = emb_a.size(0)
    emb = torch.cat([emb_a, emb_b], dim=0)  # 2xbz
    sims = emb @ emb.t()
    diag = torch.eye(sims.size(0)).to(sims.device)

    small_value = torch.tensor(-10000.).to(sims.device).to(sims.dtype)
    sims = torch.where(diag.eq(0), sims, small_value)
    gt = torch.cat([torch.arange(bz) + bz, torch.arange(bz)], dim=0).to(sims.device)
    mask = torch.cat([mask, mask], dim=0).bool()

    losses_ = F.cross_entropy((sims - diag * args.margin) / temperature, gt, reduction="none")
    loss_ = losses_[mask.bool()].mean()
    return loss_


def entropy_loss_fn(sims, mask):
    device = sims.device
    diag = torch.eye(sims.size(0)).to(device)
    local_mask = (1 - diag)
    small_value = torch.tensor(-10000.).to(device).to(sims.dtype)
    max_non_match_sim = torch.where(local_mask.bool(), sims, small_value)[mask.bool()].max(dim=1)[0]
    closest_distance = (1 / 2 - max_non_match_sim / 2).clamp(min=1e-6).sqrt()
    entropy_loss_ = -closest_distance.log().mean() * args.entropy_weight
    return entropy_loss_


def train_step(batch_data):
    vid_a, vid_b = batch_data["vid_a"], batch_data["vid_b"]
    weight = batch_data["weight"]
    bz = batch_data["img_a"].size(0)
    device = batch_data["img_a"].device

    cat_x = torch.cat([batch_data["img_a"], batch_data["img_b"]], dim=0)

    embeds = model(x=cat_x)
    embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)
    emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
    emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2*bz]

    ga_emb_a_norm, ga_emb_b_norm, ga_vid_a, ga_vid_b, ga_weight = all_gather(
        args.rank, world_size, emb_a=emb_a_norm, emb_b=emb_b_norm, vid_a=vid_a[..., None], vid_b=vid_b[..., None],
        weight=weight[..., None]
    )
    sims_norm = ga_emb_a_norm @ ga_emb_b_norm.t()

    rank_mask = torch.zeros(bz * dist.get_world_size()).to(device)
    rank_mask[args.rank*bz:(args.rank + 1)*bz] = 1

    if args.product_loss:
        match_sim = (emb_a_norm * emb_b_norm).sum(dim=1)
        ici_losses_ = (1 - match_sim).exp()
        ici_loss_ = ici_losses_.mean()
    else:
        ici_loss_ = contrast_loss_fn(ga_emb_a_norm, ga_emb_b_norm, args.t, rank_mask, ga_weight) * args.ici_weight

    entropy_loss_ = entropy_loss_fn(sims_norm, rank_mask)

    return ici_loss_, entropy_loss_


def adv_loss_fn(batch_data):
    r_at_a = args.fmg_scale * batch_data["img_a"].grad / (torch.norm(batch_data["img_a"].grad) + 1e-8)
    r_at_b = args.fmg_scale * batch_data["img_b"].grad / (torch.norm(batch_data["img_b"].grad) + 1e-8)
    batch_data["img_a"] = batch_data["img_a"].clone() + r_at_a
    batch_data["img_b"] = batch_data["img_b"].clone() + r_at_b
    ici_loss_adv_, entropy_loss_adv_ = train_step(batch_data)
    loss_adv_ = ici_loss_adv_ + entropy_loss_adv_

    return loss_adv_

def val_step(batch_data):
    vid_a, vid_b = batch_data["vid_a"], batch_data["vid_b"]
    weight = batch_data["weight"]
    bz = batch_data["img_a"].size(0)
    device = batch_data["img_a"].device

    cat_x = torch.cat([batch_data["img_a"], batch_data["img_b"]], dim=0)

    with torch.no_grad():
        embeds = model(x=cat_x)
        embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)
        emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
        emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2*bz]

        ga_emb_a_norm, ga_emb_b_norm, ga_vid_a, ga_vid_b, ga_weight = all_gather(
            args.rank, world_size, emb_a=emb_a_norm, emb_b=emb_b_norm, vid_a=vid_a[..., None], vid_b=vid_b[..., None],
            weight=weight[..., None]
        )
        sims_norm = ga_emb_a_norm @ ga_emb_b_norm.t()

        rank_mask = torch.zeros(bz * dist.get_world_size()).to(device)
        rank_mask[args.rank*bz:(args.rank + 1)*bz] = 1

        if args.product_loss:
            match_sim = (emb_a_norm * emb_b_norm).sum(dim=1)
            ici_losses_ = (1 - match_sim).exp()
            ici_loss_ = ici_losses_.mean()
        else:
            ici_loss_ = contrast_loss_fn(ga_emb_a_norm, ga_emb_b_norm, args.t, rank_mask, ga_weight) * args.ici_weight

        entropy_loss_ = entropy_loss_fn(sims_norm, rank_mask)

    return ici_loss_, entropy_loss_


global_step = 0
for _e in range(start_epoch, epochs):
    model.train()
    train_sampler.set_epoch(_e)
    epoch_loss = 0.
    num_steps = 0
    for _b, batch in enumerate(train_loader):
        num_steps += 1
        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda(args.local_rank)

        if args.do_fgm:
            batch["img_a"].requires_grad_(True)
            batch["img_b"].requires_grad_(True)

        opt.zero_grad()
        if args.fp16:
            with torch.cuda.amp.autocast():
                ici_loss, entropy_loss = train_step(batch)
                loss = ici_loss + entropy_loss
                scaler.scale(loss).backward()

                if args.do_fgm:
                    loss_adv = adv_loss_fn(batch)
                    scaler.scale(loss_adv).backward()

                scaler.step(opt)
                if args.do_ema:
                    ema.update()
                scaler.update()
        else:
            ici_loss, entropy_loss = train_step(batch)
            loss = ici_loss + entropy_loss
            loss.backward()

            if args.do_fgm:
                loss_adv = adv_loss_fn(batch)
                loss_adv.backward()

            opt.step()
            if args.do_ema:
                ema.update()
        scheduler.step()
        epoch_loss += loss.cpu().item()

        global_step += 1
        if args.rank == 0 and _b % print_freq == 0:
            logger.info('Epoch %d Batch %d / %d Loss %.3f, ICI Loss %.3f, Entropy loss %.3f.' % (
                _e, _b, len(train_loader), loss.item(), ici_loss.item(), entropy_loss.item())
            )
    epoch_loss /= num_steps
    print('Train epoch loss:', epoch_loss)
    model.eval()
    val_loss = 0.
    val_ici_loss = 0.
    val_entropy_loss = 0.
    with torch.no_grad():
        for batch in val_loader:
            for _k, _v in batch.items():
                if isinstance(_v, torch.Tensor):
                    batch[_k] = _v.cuda(args.local_rank)
            ici_loss, entropy_loss = val_step(batch)
            val_loss += ici_loss.item() + entropy_loss.item()
            val_ici_loss += ici_loss.item()
            val_entropy_loss += entropy_loss.item()

    val_loss /= len(val_loader)
    val_ici_loss /= len(val_loader)
    val_entropy_loss /= len(val_loader)

    if args.rank == 0:
        logger.info('Epoch %d Val Loss %.3f, Val ICI Loss %.3f, Val Entropy loss %.3f.' % (
            _e, val_loss, val_ici_loss, val_entropy_loss)
        )
        if args.do_ema:
            ema.apply_shadow()
        ckpt = {'state_dict': model.state_dict(), 'epoch': _e}
        torch.save(ckpt, work_dir + f'/checkpoints/{cfg.experiment_name}/{cfg.run_name}/epoch_%d.pth' % _e)
        if args.do_ema:
            ema.restore()