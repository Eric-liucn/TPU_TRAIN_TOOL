#!/bin/bash

# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/diffusers_scripts/train_gpu_text_to_image.sh -O train.sh && chmod +x train.sh

# Check for --only-train argument
ONLY_TRAIN="FALSE"
for arg in "$@"
do
  if [ "$arg" == "--only-train" ]
  then
    ONLY_TRAIN="TRUE"
    break
  fi
done


# setup logger
LOGFILE="train.log"
exec > >(tee -a ${LOGFILE} )
exec 2> >(tee -a ${LOGFILE} >&2)

# init
cd "$HOME" || exit
sudo apt-get update
sudo apt-get install -y wget python3-venv git zip unzip

# setting
export MODEL_PATH="SillyL12324/chilloutmix_100ep"
export REMOTE_DATA_PATH="gs://aiforsure_ai/datasets/girls"
export CHECKPOINT_NAME="chilloutmix_200ep_1e-4.safetensors"
export LEARNING_RATE=1e-4
export NUM_TRAIN_EPOCHS=100
export LR_SCHEDULER="constant" # constant, linear, cosine, cosine_with_restarts,
export MIXED_PRECISION="bf16"
export ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION="FALSE"
export RESUME_FROM_CHECKPOINT="FALSE"

# unfrequent setting
export RESOLUTION=512
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export GRADIENT_CHECKPOINTING="FALSE"
export SCALE_LR="TRUE"
export LR_WARMUP_STEPS=0
export SNR_GAMMA=5.0
export USE_8BIT_ADAM="FALSE"
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_WEIGHT_DECAY=1e-2
export ADAM_EPSILON=1e-8
export ALLOW_TF32="FALSE"
export USE_EMA="FALSE"
export DATALOADER_NUM_WORKERS=4
export MAX_GRAD_NORM=1
export CHECKPOINTING_STEPS=10000
export CHECKPOINTS_TOTAL_LIMIT=5
export NOISE_OFFSET=0

# default setting
export DATA_PATH="$HOME/DATA"
export OUTPUT_PATH="$HOME/OUTPUT"
export CHECKPOINT_PATH="$OUTPUT_PATH/$CHECKPOINT_NAME"

if [ "$ONLY_TRAIN" == "FALSE" ]; then
  # download datasets
  mkdir -p "$DATA_PATH"
  gsutil -m cp -r -n "$REMOTE_DATA_PATH/*" "$DATA_PATH"

  # setup output dir
  mkdir -p "$OUTPUT_PATH"

  # Setup diffusers
  cd "$HOME" || exit
  git clone https://github.com/huggingface/diffusers.git
  cd "$HOME/diffusers" || exit
  python3 -m venv .env
  source .env/bin/activate
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install git+https://github.com/huggingface/diffusers
  cd "$HOME/diffusers/examples/text_to_image" || exit
  pip install -r requirements.txt
  pip install accelerate
  pip install safetensors
  pip install omegaconf
  accelerate config
else
  cd "$HOME/diffusers" || exit
  source .env/bin/activate
fi

# Run training
cd "$HOME/diffusers/examples/text_to_image" || exit

# construct command
COMMAND = "accelerate launch train_text_to_image.py "
# --pretrained_model_name_or_path
COMMAND += "--pretrained_model_name_or_path=\"$MODEL_PATH\" "
# --dataset_name
COMMAND += "--dataset_name=\"$DATA_PATH\" "
# --output_dir
COMMAND += "--output_dir=\"$OUTPUT_PATH\" "
# --resolution
COMMAND += "--resolution=$RESOLUTION "
# --train_batch_size
COMMAND += "--train_batch_size=$TRAIN_BATCH_SIZE "
# --num_train_epochs
COMMAND += "--num_train_epochs=$NUM_TRAIN_EPOCHS "
# --gradient_accumulation_steps
COMMAND += "--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
# --gradient_checkpointing
if [ "$GRADIENT_CHECKPOINTING" = "TRUE" ]; then
  COMMAND += "--gradient_checkpointing "
fi
# --learning_rate
COMMAND += "--learning_rate=$LEARNING_RATE "
# --scale_lr
if [ "$SCALE_LR" = "TRUE" ]; then
  COMMAND += "--scale_lr "
fi
# --lr_scheduler
COMMAND += "--lr_scheduler=$LR_SCHEDULER "
# --lr_warmup_steps
COMMAND += "--lr_warmup_steps=$LR_WARMUP_STEPS "
# --snr_gamma
if [ "$SNR_GAMMA" != "FALSE" ]; then
  COMMAND += "--snr_gamma=$SNR_GAMMA "
fi
# --use_8bit_adam
if [ "$USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND += "--use_8bit_adam "
fi
# --allow_tf32
if [ "$ALLOW_TF32" = "TRUE" ]; then
  COMMAND += "--allow_tf32 "
fi
# --use_ema
if [ "$USE_EMA" = "TRUE" ]; then
  COMMAND += "--use_ema "
fi
# --dataloader_num_workers
COMMAND += "--dataloader_num_workers=$DATALOADER_NUM_WORKERS "
# --adam_beta1
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND += "--adam_beta1=$ADAM_BETA1 "
fi
# --adam_beta2
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND += "--adam_beta2=$ADAM_BETA2 "
fi
# --adam_weight_decay
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND += "--adam_weight_decay=$ADAM_WEIGHT_DECAY "
fi
# --adam_epsilon
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND += "--adam_epsilon=$ADAM_EPSILON "
fi
# --max_grad_norm
COMMAND += "--max_grad_norm=$MAX_GRAD_NORM "
# --mixed_precision
if [ "$MIXED_PRECISION" != "FALSE" ]; then
  COMMAND += "--mixed_precision=$MIXED_PRECISION "
fi
# --enable_xformers_memory_efficient_attention
if [ "$ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION" = "TRUE" ]; then
  COMMAND += "--enable_xformers_memory_efficient_attention "
fi
# --resume_from_checkpoint
if [ "$RESUME_FROM_CHECKPOINT" = "TRUE" ]; then
  COMMAND += "--resume_from_checkpoint=$CHECKPOINT_PATH "
fi
# --checkpointing_steps
COMMAND += "--checkpointing_steps=$CHECKPOINTING_STEPS "
# --checkpoints_total_limit
COMMAND += "--checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT "
# --noise_offset
COMMAND += "--noise_offset=$NOISE_OFFSET "

# log command
echo COMMAND

# run command
eval COMMAND


# gen checkpoints
cd "$HOME/diffusers/scripts" || exit
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path "$OUTPUT_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --use_safetensors



