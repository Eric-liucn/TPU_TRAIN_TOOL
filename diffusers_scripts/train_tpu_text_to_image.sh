#!/bin/bash

# wget --no-cache https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/diffusers_scripts/train_tpu_text_to_image.sh -O train.sh && chmod +x train.sh

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
export MODEL_PATH="SillyL12324/chilloutmix_100ep_flax"
export REMOTE_DATA_PATH="gs://aiforsure_ai/datasets/girls"
export CHECKPOINT_NAME="chilloutmix_200ep_1e-4.safetensors"
export LEARNING_RATE=1e-4
export NUM_TRAIN_EPOCHS=10
export LR_SCHEDULER="constant" # constant, linear, cosine, cosine_with_restarts,
export MIXED_PRECISION="bf16"
export ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION="FALSE"
export RESUME_FROM_CHECKPOINT="FALSE"

# unfrequent setting
export RESOLUTION=512
export TRAIN_BATCH_SIZE=8
export SCALE_LR="TRUE"
export USE_8BIT_ADAM="FALSE"
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_WEIGHT_DECAY=1e-2
export ADAM_EPSILON=1e-8
export ALLOW_TF32="FALSE"
export USE_EMA="FALSE"
export MAX_GRAD_NORM=1
export RESUME_CHECKPOINT_PATH=""

# default setting
export DATA_PATH="$HOME/DATA"
export OUTPUT_PATH="$HOME/OUTPUT"
export OUTPUT_FLAX_PATH="$OUTPUT_PATH/FLAX"
export OUTPUT_PT_PATH="$OUTPUT_PATH/PT"
export CHECKPOINT_PATH="$OUTPUT_PATH/$CHECKPOINT_NAME"

if [ "$ONLY_TRAIN" == "FALSE" ]; then
  # download datasets
  mkdir -p "$DATA_PATH"
  gsutil -m cp -r -n "$REMOTE_DATA_PATH/*" "$DATA_PATH"

  # setup output dir
  mkdir -p "$OUTPUT_PATH"
  mkdir -p "$OUTPUT_FLAX_PATH"
  mkdir -p "$OUTPUT_PT_PATH"

  # Setup diffusers
  cd "$HOME" || exit
  git clone https://github.com/huggingface/diffusers.git
  cd "$HOME/diffusers" || exit
  python3 -m venv .env
  source .env/bin/activate
  pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
  pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-2.0-cp38-cp38-linux_x86_64.whl
    pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-2.0-cp38-cp38-linux_x86_64.whl
  pip install git+https://github.com/huggingface/diffusers
  cd "$HOME/diffusers/examples/text_to_image" || exit
  pip install -r requirements_flax.txt
  # pip install accelerate
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
COMMAND="python train_text_to_image_flax.py "
# --pretrained_model_name_or_path
COMMAND="$COMMAND--pretrained_model_name_or_path=\"$MODEL_PATH\" "
# --dataset_name
COMMAND="$COMMAND--dataset_name=\"$DATA_PATH\" "
# --output_dir
COMMAND="$COMMAND--output_dir=\"$OUTPUT_FLAX_PATH\" "
# --resolution
COMMAND="$COMMAND--resolution=$RESOLUTION "
# --train_batch_size
COMMAND="$COMMAND--train_batch_size=$TRAIN_BATCH_SIZE "
# --num_train_epochs
COMMAND="$COMMAND--num_train_epochs=$NUM_TRAIN_EPOCHS "
# --learning_rate
COMMAND="$COMMAND--learning_rate=$LEARNING_RATE "
# --scale_lr
if [ "$SCALE_LR" = "TRUE" ]; then
  COMMAND="$COMMAND--scale_lr "
fi
# --lr_scheduler
COMMAND="$COMMAND--lr_scheduler=$LR_SCHEDULER "
# --use_8bit_adam
if [ "$USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND="$COMMAND--use_8bit_adam "
fi
# --allow_tf32
if [ "$ALLOW_TF32" = "TRUE" ]; then
  COMMAND="$COMMAND--allow_tf32 "
fi
# --use_ema
if [ "$USE_EMA" = "TRUE" ]; then
  COMMAND="$COMMAND--use_ema "
fi
# --adam_beta1
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND="$COMMAND--adam_beta1=$ADAM_BETA1 "
fi
# --adam_beta2
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND="$COMMAND--adam_beta2=$ADAM_BETA2 "
fi
# --adam_weight_decay
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND="$COMMAND--adam_weight_decay=$ADAM_WEIGHT_DECAY "
fi
# --adam_epsilon
if [ "USE_8BIT_ADAM" = "TRUE" ]; then
  COMMAND="$COMMAND--adam_epsilon=$ADAM_EPSILON "
fi
# --max_grad_norm
COMMAND="$COMMAND--max_grad_norm=$MAX_GRAD_NORM "
# --mixed_precision
if [ "$MIXED_PRECISION" != "FALSE" ]; then
  COMMAND="$COMMAND--mixed_precision=$MIXED_PRECISION "
fi
# --enable_xformers_memory_efficient_attention
if [ "$ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION" = "TRUE" ]; then
  COMMAND="$COMMAND--enable_xformers_memory_efficient_attention "
fi
# --resume_from_checkpoint
if [ "$RESUME_FROM_CHECKPOINT" = "TRUE" ]; then
  COMMAND="$COMMAND--resume_from_checkpoint=$RESUME_CHECKPOINT_PATH "
fi

# log command
echo $COMMAND

# run command
eval $COMMAND

# get convert_flax_pt.py
cd "$HOME"
wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/convert_tools/convert_flax_pt.py -O convert_flax_pt.py

python3 convert_flax_pt.py fp "$OUTPUT_FLAX_PATH" "$OUTPUT_PT_PATH"

# gen checkpoints
cd "$HOME/diffusers/scripts" || exit
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path "$OUTPUT_PT_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --use_safetensors