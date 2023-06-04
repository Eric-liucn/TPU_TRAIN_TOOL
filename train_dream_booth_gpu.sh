#!/bin/bash

# download command comment
# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/train_dream_booth_gpu.sh && chmod +x train_dream_booth_gpu.sh && ./train_dream_booth_gpu.sh
cd "$HOME" || exit

LOGFILE="train.log"

# Redirect standard output and standard error to the log file
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# setup environment
sudo apt-get update
sudo apt-get install -y wget
sudo apt-get install -y python3-venv


# setup parameters
# edit these parameters to change the model and training settings
export INSTANCE_DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/kunkun/processed"
export CLASS_DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/kunkun/class_image"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_FILE_NAME="stable-diffusion-2-1-base"
export LEARNING_RATE=5e-6
export STEP=800
export INSTANCE_PROMPT="a kunkun toy figure"
export CLASS_PROMPT="a toy figure"
export NUM_CLASS_IMAGES=145

# no need to change these parameters
export INSTANCE_DATA_LOCAL_PATH="$HOME/instance_data"
export CLASS_DATA_LOCAL_PATH="$HOME/class_data"
export OUTPUT_REMOTE_PATH="gs://aiforsure_ai/train_output/dreambooth/"${MODEL_FILE_NAME?"MODEL_FILE_NAME"}"_"${LEARNING_RATE?"LEARNING_RATE"}"_"${STEP?"STEP"}
export OUTPUT_LOCAL_PATH="$HOME/OUTPUT"
export OUTPUT_LOCAL_PT_PATH="$HOME/OUTPUT/PT"
export OUTPUT_LOCAL_FLAX_PATH="$HOME/OUTPUT/FLAX"
export OUTPUT_CHECKPOINT_PATH="$OUTPUT_LOCAL_PATH/"${MODEL_FILE_NAME?"MODEL_FILE_NAME"}"_"${LEARNING_RATE?"LEARNING_RATE"}"_"${STEP?"STEP"}.safetensors

# download datasets
mkdir -p "$INSTANCE_DATA_LOCAL_PATH"
gsutil -m cp -r "$INSTANCE_DATA_REMOTE_PATH/*" "$INSTANCE_DATA_LOCAL_PATH"
mkdir -p "$CLASS_DATA_LOCAL_PATH"
gsutil -m cp -r "$CLASS_DATA_REMOTE_PATH/*" "$CLASS_DATA_LOCAL_PATH"

# setup output dir
mkdir -p "$OUTPUT_LOCAL_PATH"
mkdir -p "$OUTPUT_LOCAL_PT_PATH"
mkdir -p "$OUTPUT_LOCAL_FLAX_PATH"

# Setup diffusers
cd "$HOME" || exit
git clone https://github.com/huggingface/diffusers.git
cd "$HOME/diffusers" || exit
python3 -m venv .env
source .env/bin/activate 
pip install git+https://github.com/huggingface/diffusers
cd "$HOME/diffusers/examples/dreambooth" || exit
pip install -r requirements.txt
pip install accelerate
pip install safetensors
pip install omegaconf
# pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
accelerate config default

accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --train_text_encoder \
    --instance_data_dir="$INSTANCE_DATA_LOCAL_PATH" \
    --class_data_dir="$CLASS_DATA_LOCAL_PATH" \
    --output_dir="$OUTPUT_LOCAL_PT_PATH" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    --resolution=512 \
    --train_batch_size=1 \
    --mixed_precision=bf16 \
    --learning_rate="$LEARNING_RATE" \
    --num_class_images="$NUM_CLASS_IMAGES" \
    --max_train_steps="$STEP"

# convert diffuser model to original stable diffusion model
cd "$HOME/diffusers/scripts" || exit
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path "$OUTPUT_LOCAL_PT_PATH" \
  --checkpoint_path "$OUTPUT_CHECKPOINT_PATH" \
  --use_safetensors

# upload output to google cloud storage
gsutil -m cp -r "$OUTPUT_CHECKPOINT_PATH" "$OUTPUT_REMOTE_PATH"
