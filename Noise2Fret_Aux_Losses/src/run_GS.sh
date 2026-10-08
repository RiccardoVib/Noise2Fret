#!/bin/bash

DATA_DIR="./data"
MODEL_PATH="./models/"
BASE_CHANNELS=64
EMBED_DIM=32
BATCH_SIZE=128
EPOCHS=1000
LR=3e-4
LOSSES_STR=[""]

TRAIN_NOISE_STEPS=20   # T used during training (defines the noise schedule)
INFER_NOISE_STEPS=1   # sampling steps; may differ (e.g. DDIM / fewer steps)
TRAIN_MODEL=True       # True = train, False = inference only

if [ "$TRAIN_MODEL" = "True" ]; then
  NOISE_STEPS=$TRAIN_NOISE_STEPS
else
  NOISE_STEPS=$INFER_NOISE_STEPS
fi

python diffusion_training_GS.py \
  --data_dir $DATA_DIR \
  --model_path $MODEL_PATH \
  --noise_steps $NOISE_STEPS \
  --base_channels $BASE_CHANNELS \
  --embed_dim $EMBED_DIM \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --losses_str $LOSSES_STR \
  --train_model $TRAIN_MODEL
