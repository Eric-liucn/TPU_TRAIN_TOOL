#!/bin/bash

cd "$HOME" || exit

LOGFILE="train.log"

# Redirect standard output and standard error to the log file
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# setup nvidia cuda
sudo apt-get update
sudo apt-get install -y wget
sudo apt-get install -y python3-venv

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


# download datasets from google cloud storage
export INSTANCE_DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/kunkun/processed"
export INSTANCE_DATA_LOCAL_PATH="$HOME/instance_data"
export CLASS_DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/kunkun/class_image"
export CLASS_DATA_LOCAL_PATH="$HOME/class_data"
export MODEL_REMOTE_PATH="gs://aiforsure_ai/models/dreamlike2/diffusion_model_flax/*"
export MODEL_LOCAL_PATH="$HOME/PRETRAINED_MODEL"
export OUTPUT_REMOTE_PATH="gs://aiforsure_ai/train_output/dreamboot/sd2_1_base_kunkun_2e-5_800"
export OUTPUT_LOCAL_PATH="$HOME/OUTPUT"
export OUTPUT_LOCAL_PT_PATH="$HOME/OUTPUT/PT"
export OUTPUT_LOCAL_FLAX_PATH="$HOME/OUTPUT/FLAX"

# download datasets
mkdir -p "$INSTANCE_DATA_LOCAL_PATH"
gsutil -m cp -r "$INSTANCE_DATA_REMOTE_PATH" "$INSTANCE_DATA_LOCAL_PATH/../"
mv "$INSTANCE_DATA_LOCAL_PATH/../processed" "$INSTANCE_DATA_LOCAL_PATH"
mkdir -p "$CLASS_DATA_LOCAL_PATH"
gsutil -m cp -r "$CLASS_DATA_REMOTE_PATH" "$CLASS_DATA_LOCAL_PATH/../"
mv "$CLASS_DATA_LOCAL_PATH/../class_image" "$CLASS_DATA_LOCAL_PATH"

# download model
# pass

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
cd "$HOME/diffusers/examples/dreambooth" || exit
pip install -r requirements.txt
pip install accelerate
pip install safetensors
pip install omegaconf
accelerate config default


accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --train_text_encoder \
    --instance_data_dir="$INSTANCE_DATA_LOCAL_PATH" \
    --class_data_dir="$CLASS_DATA_LOCAL_PATH" \
    --output_dir="$OUTPUT_LOCAL_PT_PATH" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a kunkun toy figure" \
    --class_prompt="a toy figure" \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=2e-5 \
    --num_class_images=150 \
    --max_train_steps=800

# upload output to google cloud storage
gsutil -m cp -r "$OUTPUT_LOCAL_PATH" "$OUTPUT_REMOTE_PATH"
