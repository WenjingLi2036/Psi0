#!/bin/bash

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"
export CHECKPOINT_DIR="./.runs/finetune/move-white.real.flow1000.cosine.lr1.0e-04.b128.gpus1.2604211655/"
export CHECKPOINT_STEP=25000

python src/psi/deploy/psi_serve_rtc-trainingtimertc.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP} \