#!/bin/bash


module load gcc
module load conda
conda activate /global/common/software/m4629/environments/netfound

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

# copy the file and modify the params as needed
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
    --save_strategy epoch \
    --do_eval \
    --validation_split_percentage 20 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 1 \
    --train_dir /global/homes/h/haarika/pscratch/test_directories/speedtest_debug/tokens \
    --output_dir /global/homes/h/haarika/pscratch/network-data-representation/model_output \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --mlm_probability 0.20 \
    --max_train_samples 10000 \
    --deepspeed /global/homes/h/haarika/pscratch/network-data-representation/src/train/deepspeed_stage2.json
    '
