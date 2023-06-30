#!/bin/ssh
# wget https://github.com/Eric-liucn/TPU_TRAIN_TOOL/raw/main/kohya_scripts/train_gpu_lora.sh && chmod +x train_gpu_lora.sh && screen -S train ./train_gpu_lora.sh

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
export DATASET_REPEATS=1
export SAVE_PRECISION="fp16"
export SAVE_EVERY_N_EPOCHS=10
export MEM_EFF_ATTN=FALSE
export XFORMERS=TRUE
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
export TRAIN_DATA_PATH="$DATA_PATH/train"
export REG_DATA_PATH="$DATA_PATH/reg"
export OUTPUT_PATH="$HOME/OUTPUT"
export MODEL_PATH="$HOME/MODEL/model.safetensors"
export NETWORK_MODULE="networks.lora"

# create dirs and download data and model
mkdir -p "$DATA_PATH"
mkdir -p "$TRAIN_DATA_PATH"
mkdir -p "$REG_DATA_PATH"
mkdir -p "$OUTPUT_PATH"
mkdir -p "$MODEL_PATH"

# first download data file and extract, copy all images(.png, .jpg, .jpeg) and txt(.txt) files to $DATA_PATH
gsutil -m cp "$DATA_REMOTE_PATH" "/tmp/data.zip"
unzip "/tmp/data.zip" -d "/tmp/data"
# check if train dir exists, if not exit with error
if [ ! -d "/tmp/data/train" ]; then
  echo "train dir not exists"
  exit 1
fi
# check if reg dir exists, if not exit warning
if [ ! -d "/tmp/data/reg" ]; then
  echo "reg dir not exists, will not use reg data"
  export REG_DATA_PATH=NONE
fi

find "/tmp/data/train" -type f -name {*.png,*.jpg,*.jpeg,*.txt} -exec cp {} "$TRAIN_DATA_PATH" \;
# if reg dir exists, copy reg data
if [ -d "/tmp/data/reg" ]; then
  find "/tmp/data/reg" -type f -name {*.png,*.jpg,*.jpeg,*.txt} -exec cp {} "$REG_DATA_PATH" \;
fi

# download model file
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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers, pillow

# download caption script and do caption
wget wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/caption_tools/gen_captions.py -O gen_captions.py
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
if [ ! -d "$HOME/train_env" ]; then
  python3 -m venv train_env
else
  rm -rf train_env
  python3 -m venv train_env
fi

# activate train env
source "$HOME/train_env/bin/activate"
pip3 install -r requirements.txt
accelerate config default

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
COMMAND="$COMMAND --train_data_dir=\"$TRAIN_DATA_PATH\" "
COMMAND="$COMMAND --caption_extension=\".txt\" "
COMMAND="$COMMAND --resolution=\"$RESOLUTION\" "
COMMAND="$COMMAND --cache_latents_to_disk "
if [ "$REG_DATA_PATH" != NONE ]; then
    COMMAND="$COMMAND --reg_data_dir=\"$REG_DATA_PATH\" "
fi
COMMAND="$COMMAND --dataset_repeats=\"$DATASET_REPEATS\" "
COMMAND="$COMMAND --save_precision=\"$SAVE_PRECISION\" "
COMMAND="$COMMAND --save_every_n_epochs=\"$SAVE_EVERY_N_EPOCHS\" "
COMMAND="$COMMAND --batch_size=\"$TRAIN_BATCH_SIZE\" "
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
COMMAND="$COMMAND --max_epochs=\"$MAX_TRAIN_EPOCHS\" "
COMMAND="$COMMAND --max_data_loader_n_workers=\"$MAX_DATA_LOADER_N_WORKERS\" "
if [ "$GRADIENT_CHECKPOINTING" = TRUE ]; then
    COMMAND="$COMMAND --gradient_checkpointing "
fi
if [ "$GRADIENT_CHECKPOINTING" = TRUE ]; then
    COMMAND="$COMMAND --gradient_accumulation_steps=\"$GRADIENT_ACCUMULATION_STEPS\" "
fi
if [ !"$MIXED_PRECISION" = "FALSE" ]; then
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
COMMAND="$COMMAND --SAVE_MODEL_AS=\"$SAVE_MODEL_AS\" "
COMMAND="$COMMAND --network_module=\"$NETWORK_MODULE\" "
COMMAND="$COMMAND --output_dir=\"$OUTPUT_PATH\" "
COMMAND="$COMMAND --output_name=\"$OUTPUT_NAME\" "



