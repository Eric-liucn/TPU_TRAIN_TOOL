#!/bin/bash

LOGFILE="train.log"

# Redirect standard output and standard error to the log file
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# init
cd "$HOME" || exit
sudo apt-get update
sudo apt-get install -y wget
sudo apt-get install -y python3-venv
wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/convert_flax_pt.py

export DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/lady/train"
export DATA_LOCAL_PATH="$HOME/DATA"

export MODEL_REMOTE_PATH="gs://aiforsure_ai/models/dreamlike2/diffusion_model_flax/*"
export MODEL_LOCAL_PATH="$HOME/PRETRAINED_MODEL"

export OUTPUT_REMOTE_PATH="gs://aiforsure_ai/train_output/text_to_img/dreamlike2_lr1e-5_15000"
export OUTPUT_LOCAL_PT_PATH="$HOME/OUTPUT/PT"
export OUTPUT_LOCAL_FLAX_PATH="$HOME/OUTPUT/FLAX"
export OUTPUT_CHECKPOINT_PATH="$HOME/OUTPUT/dreamlike2_lr1e-5_15000.safetensors"
# download datasets
mkdir -p "$DATA_LOCAL_PATH"
gsutil -m cp -r "$DATA_REMOTE_PATH" "$DATA_LOCAL_PATH"
# download model
mkdir -p "$MODEL_LOCAL_PATH"
gsutil -m cp -r "$MODEL_REMOTE_PATH" "$MODEL_LOCAL_PATH"
# setup output dir
mkdir -p "$OUTPUT_LOCAL_PT_PATH"
mkdir -p "$OUTPUT_LOCAL_FLAX_PATH"

# Setup diffusers
cd "$HOME" || exit
git clone https://github.com/huggingface/diffusers.git
cd "$HOME/diffusers" || exit
python3 -m venv .env
source .env/bin/activate
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/huggingface/diffusers
cd "$HOME/diffusers/examples/text_to_image" || exit
pip install -r requirements_flax.txt
pip install accelerate
pip install safetensors
pip install omegaconf
accelerate config default

# Run training
cd "$HOME/diffusers/examples/text_to_image" || exit

python train_text_to_image_flax.py \
  --pretrained_model_name_or_path="$MODEL_LOCAL_PATH" \
  --dataset_name="$DATA_LOCAL_PATH" \
  --resolution=512 \
  --mixed_precision=bf16 \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-5 \
  --output_dir="$OUTPUT_LOCAL_FLAX_PATH"

cd "$HOME" || exit
python convert_flax_pt.py fp "$OUTPUT_LOCAL_FLAX_PATH" "$OUTPUT_LOCAL_PT_PATH"


cd "$HOME/diffusers/scripts" || exit
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path "$OUTPUT_LOCAL_PT_PATH" \
  --checkpoint_path "$OUTPUT_CHECKPOINT_PATH" \
  --use_safetensors
gsutil -m cp -r "$HOME/OUTPUT/*" "$OUTPUT_REMOTE_PATH"

gcloud compute tpus tpu-vm delete train --zone=us-central1-b --quiet