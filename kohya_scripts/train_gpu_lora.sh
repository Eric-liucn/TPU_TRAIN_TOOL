#!/bin/bash
# wget https://github.com/Eric-liucn/TPU_TRAIN_TOOL/raw/main/kohya_scripts/train_gpu_lora.sh && chmod +x train_gpu_lora.sh && screen -S train ./train_gpu_lora.sh

# install some packages
sudo apt-get update
sudo apt-get install -y git
sudo apt-get install -y python3-venv
sudo apt-get install -y screen

# settings
export OUTPUT_NAME="haneame"
export DATA_REMOTE_PATH="gs://aiforsure_ai/zip_datasets/haneame.zip"
export CAPTION_METHOD="blip_large" # git_large_coco, git_large_textcaps, blip_base, blip_large, blip2, vitgpt
export MODEL_REMOTE_PATH="gs://aiforsure_ai/train_output/text_to_img/chilloutmix-100k_lr5e-6_100epochs/chilloutmix-100k_lr5e-6_100epochs.safetensors"
export TRAIN_BATCH_SIZE=1
export MAX_TRAIN_EPOCHS=100
export LEARNING_RATE=1e-4
export LR_SCHEDULER="cosine" # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor

export V2_MODEL=FALSE
export RESOLUTION=512
export SAVE_PRECISION="fp16"
export SAVE_EVERY_N_EPOCHS=10
export MEM_EFF_ATTN=FALSE
export XFORMERS=FALSE
export VAE=FALSE
export MAX_DATA_LOADER_N_WORKERS=4
export GRADIENT_CHECKPOINTING=TRUE
export GRADIENT_ACCUMULATION_STEPS=1
export MIXED_PRECISION="fp16"
export USE_8BIT_ADAM=TRUE
export LR_WARMUP_STEPS=0
export SAVE_MODEL_AS="safetensors"

# default settings
export DATA_PATH="$HOME/DATA"
export REG_DATA_PATH="$HOME/REG_DATA"
export OUTPUT_PATH="$HOME/OUTPUT"
export MODEL_PATH="$HOME/MODEL/model.safetensors"
export NETWORK_MODULE="networks.lora"

# create dirs and download data and model
mkdir -p "$DATA_PATH"
mkdir -p "$REG_DATA_PATH"
mkdir -p "$OUTPUT_PATH"

# first download data file and extract, copy all images(.png, .jpg, .jpeg) and txt(.txt) files to $DATA_PATH
gsutil -m cp "$DATA_REMOTE_PATH" "/tmp/data.zip"
unzip "/tmp/data.zip" -d "/tmp/data"

# check if reg dir exists, if not exit warning
if [ ! -d "/tmp/data/reg" ]; then
  echo "reg dir not exists, will not use reg data"
  export REG_DATA_PATH=FALSE
else
  echo "reg dir exists, will use reg data"
  find "/tmp/data/reg" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.txt" \) -exec cp {} "$REG_DATA_PATH" \;
fi

# copy all sub directories in /tmp/data exclude 'reg' to $DATA_PATH
find "/tmp/data" -mindepth 1 -maxdepth 1 -type d ! -name "reg" -exec cp -r {} "$DATA_PATH" \;

# download model file to data folder
gsutil -m cp "$MODEL_REMOTE_PATH" "$MODEL_PATH"

# prepare caption env
# check if "$HOME/caption_env" exists, if not create it, if exists, delete it and create it
if [ ! -d "$HOME/caption_env" ]; then
  cd "$HOME" || exit
  python3 -m venv caption_env
else
  cd "$HOME" || exit
  rm -rf caption_env
  python3 -m venv caption_env
fi

# activate caption env
source "$HOME/caption_env/bin/activate"
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install transformers
pip3 install pillow
pip3 install tqdm

# download caption script and do caption
wget wget https://github.com/Eric-liucn/TPU_TRAIN_TOOL/raw/main/caption_tools/gen_captions.py -O gen_captions.py
python3 gen_captions.py --data_path="$DATA_PATH" --model_str="$CAPTION_METHOD"
deactivate

# clone https://github.com/kohya-ss/sd-scripts.git if not exists, else pull
if [ ! -d "$HOME/sd-scripts" ]; then
  cd "$HOME" || exit
  git clone https://github.com/kohya-ss/sd-scripts.git
else
    cd "$HOME/sd-scripts" || exit
    git pull
fi

# create train env and install requirments
cd "$HOME/sd-scripts" || exit
if [ ! -d "$HOME/sd-scripts/train_env" ]; then
  python3 -m venv train_env
else
  rm -rf train_env
  python3 -m venv train_env
fi

# activate train env
source "$HOME/sd-scripts/train_env/bin/activate"
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install --upgrade -r requirements.txt
pip3 install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/linux/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl
accelerate config

# train model
# if V2_MODEL is TRUE, add --v2, else not add
# if REG_DATA_PATH is not empty, add --reg_data_dir="$REG_DATA_PATH", else not add
cd "$HOME/sd-scripts" || exit

# construct command
COMMAND="accelerate launch --num_cpu_threads_per_process 4 train_network.py "
if [ "$V2_MODEL" = TRUE ]; then
    COMMAND="$COMMAND --v2 "
fi
COMMAND="$COMMAND --pretrained_model_name_or_path=\"$MODEL_PATH\" "
COMMAND="$COMMAND --train_data_dir=\"$DATA_PATH\" "
COMMAND="$COMMAND --caption_extension=\".txt\" "
COMMAND="$COMMAND --resolution=\"$RESOLUTION\" "
COMMAND="$COMMAND --cache_latents_to_disk "
if [ "$REG_DATA_PATH" != FALSE ]; then
    COMMAND="$COMMAND --reg_data_dir=\"$REG_DATA_PATH\" "
fi
COMMAND="$COMMAND --save_precision=\"$SAVE_PRECISION\" "
COMMAND="$COMMAND --save_every_n_epochs=\"$SAVE_EVERY_N_EPOCHS\" "
COMMAND="$COMMAND --train_batch_size=\"$TRAIN_BATCH_SIZE\" "
if [ "$MEM_EFF_ATTN" = TRUE ]; then
    COMMAND="$COMMAND --mem_eff_attn "
fi
if [ "$XFORMERS" = TRUE ]; then
    COMMAND="$COMMAND --xformers "
fi
# if vae not FALSE, add --vae
if [ "$VAE" != FALSE ]; then
    COMMAND="$COMMAND --vae=\"$VAE\" "
fi
COMMAND="$COMMAND --max_train_epochs=\"$MAX_TRAIN_EPOCHS\" "
COMMAND="$COMMAND --max_data_loader_n_workers=\"$MAX_DATA_LOADER_N_WORKERS\" "
if [ "$GRADIENT_CHECKPOINTING" = TRUE ]; then
    COMMAND="$COMMAND --gradient_checkpointing "
fi
if [ "$GRADIENT_CHECKPOINTING" = TRUE ]; then
    COMMAND="$COMMAND --gradient_accumulation_steps=\"$GRADIENT_ACCUMULATION_STEPS\" "
fi
if [ "$MIXED_PRECISION" != "FALSE" ]; then
    COMMAND="$COMMAND --mixed_precision=\"$MIXED_PRECISION\" "
fi
if [ "$USE_8BIT_ADAM" = TRUE ]; then
    COMMAND="$COMMAND --use_8bit_adam "
fi
COMMAND="$COMMAND --learning_rate=\"$LEARNING_RATE\" "
COMMAND="$COMMAND --lr_scheduler=\"$LR_SCHEDULER\" "
if [ "$LR_WARMUP_STEPS" != 0 ]; then
    COMMAND="$COMMAND --lr_warmup_steps=\"$LR_WARMUP_STEPS\" "
fi
COMMAND="$COMMAND --save_model_as=\"$SAVE_MODEL_AS\" "
COMMAND="$COMMAND --network_module=\"$NETWORK_MODULE\" "
COMMAND="$COMMAND --output_dir=\"$OUTPUT_PATH\" "
COMMAND="$COMMAND --output_name=\"$OUTPUT_NAME\" "

# run command
eval "$COMMAND"



