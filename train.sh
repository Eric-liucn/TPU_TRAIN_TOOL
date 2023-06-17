#!/bin/bash

LOGFILE="train.log"

# Redirect standard output and standard error to the log file
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# Variables to modify
DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/lady/train"
MODEL_REMOTE_PATH="gs://aiforsure_ai/models/dreamlike2/diffusion_model_flax/*"
MODEL_NAME="dreamlike2"
LEARNING_RATE=1e-5
TRAIN_STEPS=15000

# Other variables
DATA_LOCAL_PATH="$HOME/DATA"
MODEL_LOCAL_PATH="$HOME/PRETRAINED_MODEL"
OUTPUT_LOCAL_PT_PATH="$HOME/OUTPUT/PT"
OUTPUT_LOCAL_FLAX_PATH="$HOME/OUTPUT/FLAX"
OUTPUT_REMOTE_PATH="gs://aiforsure_ai/train_output/text_to_img/${MODEL_NAME}_lr${LEARNING_RATE}_${TRAIN_STEPS}"
OUTPUT_CHECKPOINT_PATH="$HOME/OUTPUT/${MODEL_NAME}_lr${LEARNING_RATE}_${TRAIN_STEPS}.safetensors"

# Functions
setup_environment() {
  cd "$HOME" || exit
  sudo apt-get update
  sudo apt-get install -y wget python3-venv
  wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/convert_flax_pt.py
}

download_data_and_model() {
  mkdir -p "$DATA_LOCAL_PATH" "$MODEL_LOCAL_PATH"
  gsutil -m cp -r "$DATA_REMOTE_PATH" "$DATA_LOCAL_PATH"
  gsutil -m cp -r "$MODEL_REMOTE_PATH" "$MODEL_LOCAL_PATH"
}

setup_output_dirs() {
  mkdir -p "$OUTPUT_LOCAL_PT_PATH" "$OUTPUT_LOCAL_FLAX_PATH"
}

setup_diffusers() {
  cd "$HOME" || exit
  git clone https://github.com/huggingface/diffusers.git
  cd "$HOME/diffusers" || exit
  python3 -m venv .env
  source .env/bin/activate
  pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install git+https://github.com/huggingface/diffusers
  cd "$HOME/diffusers/examples/text_to_image" || exit
  pip install -r requirements_flax.txt accelerate safetensors omegaconf
  accelerate config default
}

run_training() {
  cd "$HOME/diffusers/examples/text_to_image" || exit

  python train_text_to_image_flax.py \
    --pretrained_model_name_or_path="$MODEL_LOCAL_PATH" \
    --dataset_name="$DATA_LOCAL_PATH" \
    --resolution=512 \
    --mixed_precision=bf16 \
    --train_batch_size=1 \
    --max_train_steps="$TRAIN_STEPS" \
    --learning_rate="$LEARNING_RATE" \
    --output_dir="$OUTPUT_LOCAL_FLAX_PATH"
}

convert_and_upload() {
  cd "$HOME" || exit
  python convert_flax_pt.py fp "$OUTPUT_LOCAL_FLAX_PATH" "$OUTPUT_LOCAL_PT_PATH"

  cd "$HOME/diffusers/scripts" || exit
  python convert_diffusers_to_original_stable_diffusion.py \
    --model_path "$OUTPUT_LOCAL_PT_PATH" \
    --checkpoint_path "$OUTPUT_CHECKPOINT_PATH" \
    --use_safetensors
  gsutil -m cp -r "$HOME/OUTPUT/*" "$OUTPUT_REMOTE_PATH"
}

delete_tpu_vm() {
  gcloud compute tpus tpu-vm delete train --zone=us-central1-b --quiet
}

# Main
setup_environment
download_data_and_model
setup_output_dirs
setup_diffusers
run_training
convert_and_upload
delete_tpu_vm