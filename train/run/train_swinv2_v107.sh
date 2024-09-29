projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

config=configs/config_swinv2_v107.py
workdir=./

python3 -m torch.distributed.launch --nproc_per_node ${gpu_count} train.py --config $config \
--work_dir $workdir  \
--batch_size 64 \
--num_workers 4 \
--epochs 40 \
--warmup_ratio 0.05 \
--t 0.05 \
--lr 1e-4 \
--entropy_weight 30 \
--seed 95288 \
--fp16 \
--checkpointing