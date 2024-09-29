# from models import EfficientNetEmbedding
from data.dataset import KeyframeDataset
from models import load_swinv2
swinv2_checkpoint = 'checkpoints_pretrain/swinv2_v115.torchscript.pt'

experiment_name = 'swin'
run_name = 'debug'
model = load_swinv2(swinv2_checkpoint)
train_dataset = KeyframeDataset(root_dir='../zorin/train', meta_file='../zorin/train_split.csv', width=256)