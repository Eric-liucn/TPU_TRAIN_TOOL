#!/bin/bash

# Setup gcsfuse
gcloud init
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install fuse gcsfuse
gcsfuse -v

gcloud auth application-default login
mkdir "$HOME/storage"
gcsfuse --implicit-dirs aiforsure_ai "$HOME/storage"

# Setup diffusers
git clone https://github.com/huggingface/diffusers.git
cd "$HOME/diffusers" || exit
python3 -m venv .env
source .env/bin/activate
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/huggingface/diffusers
cd "$HOME/diffusers/examples/text_to_image" || exit
pip install -r requirements_flax.txt

# Run training
cd "$HOME/diffusers/examples/text_to_image" || exit
export MODEL_PATH="$HOME/storage/models/basil_mix/diffusion_model_flax"
export DATA_PATH="$HOME/storage/datasets/lady"
export OUTPUT_PATH="$HOME/storage/train_output/text_to_img/basil_lr5e-6_30000/flax"
export OUTPUT_CONVERT_PATH="$HOME/storage/train_output/text_to_img/basil_lr5e-6_30000/pt"

python train_text_to_image_flax.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --dataset_name="$DATA_PATH" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=30000 \
  --learning_rate=5e-6 \
  --output_dir="$OUTPUT_PATH"

mkdir -p "$OUTPUT_CONVERT_PATH"

python convert_flax_pt.py fp "$OUTPUT_PATH" "$OUTPUT_CONVERT_PATH"