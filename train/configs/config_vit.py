# from models import EfficientNetEmbedding
from data.dataset import KeyframeDataset
from models import load_vit

vit_checkpoint = 'checkpoints_pretrain/vit_v68.torchscript.pt' 
experiment_name = 'vit'
run_name = 'final_train'
model = load_vit(vit_checkpoint)
train_dataset = KeyframeDataset(root_dir='../zorin/train', meta_file='../zorin/train_split.csv', width=384)
val_dataset = KeyframeDataset(root_dir='../zorin/train', meta_file='../zorin/val_split.csv', width=384)