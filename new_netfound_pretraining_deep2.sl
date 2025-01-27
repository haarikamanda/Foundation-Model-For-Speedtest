#!/bin/bash

# fixed params - do not change
#SBATCH --account=m4629_g
#SBATCH --licenses=scratch
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4

# modifiable params
#SBATCH --job-name=speedtest_pre_training
#SBATCH --time=4:00:00
#SBATCH --nodes=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=regular

module load gcc
module load conda
conda activate /global/common/software/m4629/environments/netfound

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c '\
    python \
    -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   /global/homes/h/haarika/pscratch/network-data-representation/src/train/NetfoundPretraining.py \
    --report_to tensorboard \
    --save_safetensors false \
    --bf16 \
    --do_train \
    --per_device_train_batch_size 1 \
     --dataloader_num_workers 16 \
    --save_strategy epoch \
    --do_eval \
    --validation_split_percentage 20 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 1 \
    --dataloader_prefetch_factor 8 \
    --gradient_accumulation_steps 1 \
    --train_dir /global/homes/h/haarika/pscratch/test_directories/speedtest_debug/tokens \
    --output_dir /global/homes/h/haarika/pscratch/network-data-representation/model_output \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --mlm_probability 0.20 \
    --max_train_samples 10000 \
    --deepspeed /global/homes/h/haarika/pscratch/network-data-representation/src/train/deepspeed_stage2.json
    '

srun --ntasks-per-node=1 --gpus-per-node=1 --jobid $SLURM_JOBID bash -c '\
    torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   /global/homes/h/haarika/pscratch/network-data-representation/src/train/NetfoundPretraining.py \
    --report_to tensorboard \
    --save_safetensors false \
    --bf16 \
    --do_train \
    --per_device_train_batch_size 2 \
     --dataloader_num_workers 16 \
    --save_strategy epoch \
    --do_eval \
    --validation_split_percentage 20 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 2 \
    --dataloader_prefetch_factor 8 \
    --gradient_accumulation_steps 1 \
    --train_dir /global/homes/h/haarika/pscratch/test_directories/speedtest_debug/tokens \
    --output_dir /global/homes/h/haarika/pscratch/network-data-representation/model_output \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --mlm_probability 0.20 \
    --max_train_samples 10000 \
    --deepspeed /global/homes/h/haarika/pscratch/network-data-representation/src/train/deepspeed_stage2.json \
    --resume_from_checkpoint /global/homes/h/haarika/pscratch/network-data-representation/model_output/checkpoint-18000
    '


    srun --ntasks-per-node=1 --gpus-per-node=4 --jobid $SLURM_JOBID bash -c '\
    torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   /global/homes/h/haarika/pscratch/network-data-representation/src/train/NetfoundPretraining.py \
    --report_to tensorboard \
    --save_safetensors false \
    --bf16 \
    --per_device_train_batch_size 2 \
    --dataloader_num_workers 16 \
    --save_strategy epoch \
    --train_dir /global/homes/h/haarika/pscratch/test_directories/speedtest_debug/tokens \
    --output_dir /global/homes/h/haarika/pscratch/network-data-representation/model_output2 \
    --do_eval \
    --validation_split_percentage 20 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 2 \
    --dataloader_prefetch_factor 8 \
    --learning_rate 2e-5 \
    --mlm_probability 0.20 \
    --max_eval_samples 10000 \
    --resume_from_checkpoint /global/homes/h/haarika/pscratch/network-data-representation/model_output/checkpoint-18000 \
    '


srun --ntasks-per-node=1 --gpus-per-node=4 --jobid $SLURM_JOBID bash -c '\
    torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   /global/homes/h/haarika/pscratch/network-data-representation/src/train/NetfoundPretraining.py \
    --report_to tensorboard \
    --save_safetensors false \
    --bf16 \
    --do_train \
    --per_device_train_batch_size 2 \
     --dataloader_num_workers 16 \
    --save_strategy epoch \
    --do_eval \
    --validation_split_percentage 20 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 2 \
    --dataloader_prefetch_factor 8 \
    --gradient_accumulation_steps 1 \
    --train_dir /global/homes/h/haarika/pscratch/test_directories/speedtest_debug/tokens \
    --output_dir /global/homes/h/haarika/pscratch/network-data-representation/model_output2 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --mlm_probability 0.20 \
    --deepspeed /global/homes/h/haarika/pscratch/network-data-representation/src/train/deepspeed_stage2.json \
    --resume_from_checkpoint /global/homes/h/haarika/pscratch/network-data-representation/model_output2/checkpoint-15000 \

    '