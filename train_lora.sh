#!/bin/bash
# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/train_lora.sh && chmod +x train_lora.sh && screen -dmS train ./train_lora.sh

cd "$HOME" || exit

LOGFILE="train.log"

# Redirect standard output and standard error to the log file
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# setup environment
sudo apt-get update
sudo apt-get install -y wget, git
sudo apt-get install -y python3-venv

# Variables to modify

# dataset path
REMOTE_DATA_PATH="gs://aiforsure_ai/datasets/kunkun/train"

# model setting
REMOTE_MODEL_PATH=""
HUGGINGFACE_MODEL_PATH="ram"
MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
USE_HUGGINGFACE_MODEL_PATH="True"

# training setting
LEARNING_RATE=1e-4
EPOCHS=100
INSTANCE_PROMPT="a kunkun toy figure"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
CHECKPOINT_STEP=500
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0

# download dataset and model
cd "$HOME" || exit
mkdir -p "$HOME/DATA"
gsutil -m cp -r "$REMOTE_DATA_PATH" "$HOME/DATA"

# if not use hugingface model, download model from remote path
if [ "$USE_HUGGINGFACE_MODEL_PATH" = "False" ]; then
  mkdir -p "$HOME/PRETRAINED_MODEL"
  gsutil -m cp -r "$REMOTE_MODEL_PATH" "$HOME/PRETRAINED_MODEL"
  MODEL_LOCAL_PATH="$HOME/PRETRAINED_MODEL"
else
  MODEL_LOCAL_PATH="$HUGGINGFACE_MODEL_PATH"
fi

# setup output dir
mkdir -p "$HOME/OUTPUT"
OUTPUT_DIR="$HOME/OUTPUT/${MODEL_NAME}_lr${LEARNING_RATE}_${EPOCHS}epochs"
mkdir -p "$OUTPUT_DIR"


# setup diffusers
cd "$HOME" || exit
git clone https://github.com/huggingface/diffusers.git
cd "$HOME/diffusers" || exit
python3 -m venv .env
source .env/bin/activate
pip install git+https://github.com/huggingface/diffusers
cd "$HOME/diffusers/examples/research_projects/lora || exit
pip install -r requirements.txt
pip install safetensors
pip install omegaconf
pip install accelerate
accelerate config 

# training
cd "$HOME/diffusers/examples/research_projects/lora" || exit
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_LOCAL_PATH" \
  --dataset_name=$HOME/DATA \
  --caption_column="text" \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --num_train_epochs=$EPOCHS \
  --checkpointing_steps=$CHECKPOINT_STEP \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --output_dir="$OUTPUT_DIR" \
  --use_peft \
  --lora_r=4 \
  --lora_alpha=32 \
  --lora_text_encoder_r=4 \
  --lora_text_encoder_alpha=32


