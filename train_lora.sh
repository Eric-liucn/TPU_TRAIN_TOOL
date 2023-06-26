#!/bin/bash
# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/train_lora.sh && chmod +x train_lora.sh && screen -dmS train . train_lora.sh

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
export REMOTE_DATA_PATH="gs://aiforsure_ai/datasets/kunkun/train"
export DATA_LOCAL_PATH="$HOME/DATA"

# model setting
export REMOTE_MODEL_PATH=""
export HUGGINGFACE_MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="stable-diffusion-2-1-base"
export USE_HUGGINGFACE_MODEL_PATH="True"

# training setting
export LEARNING_RATE=1e-4
export EPOCHS=100
export INSTANCE_PROMPT="a kunkun toy figure"
export RESOLUTION=512
export TRAIN_BATCH_SIZE=1
export CHECKPOINT_STEP=500
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0

# download dataset and model
cd "$HOME" || exit
mkdir -p "$HOME/DATA"
gsutil -m cp -r "$REMOTE_DATA_PATH" "$DATA_LOCAL_PATH"

# if not use hugingface model, download model from remote path
if [ "$USE_HUGGINGFACE_MODEL_PATH" = "False" ]; then
  mkdir -p "$HOME/PRETRAINED_MODEL"
  gsutil -m cp -r "$REMOTE_MODEL_PATH" "$HOME/PRETRAINED_MODEL"
  export MODEL_LOCAL_PATH="$HOME/PRETRAINED_MODEL"
else
  export MODEL_LOCAL_PATH="$HUGGINGFACE_MODEL_PATH"
fi

# setup output dir
mkdir -p "$HOME/OUTPUT"
export OUTPUT_DIR="$HOME/OUTPUT/${MODEL_NAME}_lr${LEARNING_RATE}_${EPOCHS}epochs"
mkdir -p "$OUTPUT_DIR"


# setup diffusers
cd "$HOME" || exit
git clone https://github.com/huggingface/diffusers.git
cd "$HOME/diffusers" || exit
python3 -m venv .env
source .env/bin/activate
pip install git+https://github.com/huggingface/diffusers
cd "$HOME/diffusers/examples/research_projects/lora" || exit
pip install -r requirements.txt
pip install safetensors
pip install omegaconf
pip install accelerate
accelerate config 

# training
cd "$HOME/diffusers/examples/research_projects/lora" || exit
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_LOCAL_PATH" \
  --dataset_name="$DATA_LOCAL_PATH" \
  --caption_column="text" \
  --resolution="$RESOLUTION" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --num_train_epochs="$EPOCHS" \
  --checkpointing_steps="$CHECKPOINT_STEP" \
  --learning_rate="$LEARNING_RATE" \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps="$LR_WARMUP_STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --use_peft \
  --lora_r=4 \
  --lora_alpha=32 \
  --lora_text_encoder_r=4 \
  --lora_text_encoder_alpha=32


