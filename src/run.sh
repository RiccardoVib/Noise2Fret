#!/bin/bash

DATA_DIR="./data"
MODEL_PATH="./models/my_model.pt"
NOISE_STEPS=1000
BASE_CHANNELS=64
EMBED_DIM=32
FEAT="all"
BATCH_SIZE=128
USE_PRE=False
EPOCHS=60
LR=3e-4
LOSSES_STR=[""]
TRAIN_MODEL=False

python starter.py \
  --data_dir $DATA_DIR \
  --model_path $MODEL_PATH \
  --noise_steps $NOISE_STEPS \
  --base_channels $BASE_CHANNELS \
  --embed_dim $EMBED_DIM \
  --feat $FEAT \
  --batch_size $BATCH_SIZE \
  --use_pre $USE_PRE \
  --epochs $EPOCHS \
  --lr $LR \
  --losses_str $LOSSES_STR \
  --train_model $TRAIN_MODEL