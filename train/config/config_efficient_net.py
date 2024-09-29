from models import EfficientNetEmbedding
from data.dataset import KeyframeDataset, ValKeyframeDataset

experiment_name = 'efficient_net'
run_name = 'final_train_fp32'
model = EfficientNetEmbedding(embedding_dim=128)
train_dataset = KeyframeDataset(root_dir='../zorin/train', meta_file='../zorin/train_split.csv', width=256)
val_dataset = ValKeyframeDataset(root_dir='../zorin/train', meta_file='../zorin/val_split.csv', width=256)
