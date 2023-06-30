# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/train_new.sh -O train_new.sh && chmod +x train_new.sh && ./train_new.sh
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

# setting
export DATA_REMOTE_PATH="gs://aiforsure_ai/datasets/fuli/train"
export DATA_LOCAL_PATH="$HOME/DATA"
export BASE_MODEL_REPO="SillyL12324/chilloutmix_flax"
export BASE_MODEL_NAME="chilloutmix"
export LEARNING_RATE=5e-6
export MAX_TRAIN_STEPS=100000

# setting for output
export LOCAL_OUTPUT_PATH="$HOME/OUTPUT"
export LOCAL_OUTPUT_PT_PATH="$LOCAL_OUTPUT_PATH/PT"
export LOCAL_OUTPUT_FLAX_PATH="$LOCAL_OUTPUT_PATH/FLAX"
export LOCAL_CHECKPOINT_PATH="$LOCAL_OUTPUT_PATH/$BASE_MODEL_NAME"_lr"$LEARNING_RATE"_"$MAX_TRAIN_STEPS".safetensors
export REMOTE_OUTPUT_PATH="gs://aiforsure_ai/train_output/text_to_img/"$BASE_MODEL_NAME"_lr"$LEARNING_RATE"_"$MAX_TRAIN_STEPS

# download datasets
mkdir -p "$DATA_LOCAL_PATH"
gsutil -m cp -r "$DATA_REMOTE_PATH" "$DATA_LOCAL_PATH"

# setup output dir
mkdir -p "$LOCAL_OUTPUT_PT_PATH"
mkdir -p "$LOCAL_OUTPUT_FLAX_PATH"

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
  --pretrained_model_name_or_path="$BASE_MODEL_REPO" \
  --dataset_name="$DATA_LOCAL_PATH" \
  --resolution=512 \
  --mixed_precision=bf16 \
  --train_batch_size=1 \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --learning_rate="$LEARNING_RATE" \
  --output_dir="$LOCAL_OUTPUT_FLAX_PATH" \

# convert flax to pt
cd "$HOME" || exit
python convert_flax_pt.py fp "$LOCAL_OUTPUT_FLAX_PATH" "$LOCAL_OUTPUT_PT_PATH"

# gen checkpoints
cd "$HOME/diffusers/scripts" || exit
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path "$LOCAL_OUTPUT_PT_PATH" \
  --checkpoint_path "$LOCAL_CHECKPOINT_PATH" \
  --use_safetensors
gsutil -m cp -r "$HOME/OUTPUT/*" "$REMOTE_OUTPUT_PATH"