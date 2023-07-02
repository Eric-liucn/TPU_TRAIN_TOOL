#!/bin/bash

# this script only works with setting up train instance...

# settings
export JOB_ID="000001"
export MODEL="chilloutmix_100ep"
export DATA_REMOTE_PATH="gs://aiforsure_ai/zip_datasets/jjj.zip"
export INSTANCE_PROMPT="jjj 1girl"
export CLASS_PROMPT="1girl"
export NUM_TRAIN_EPOCHS=100
export LEARNING_RATE=1e-4
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
export LR_SCHEDULER="constant"
export MIXED_PRECISION="fp16"
export ENABLE_XFORMERS=TRUE


export WITH_PRIOR_PRESERVATION=TRUE
export PRIOR_LOSS_WEIGHT=1.0
export NUM_CLASS_IMAGES=100
export RESOLUTION=512
export CENTER_CROP=FALSE
export TRAIN_TEXT_ENCODER=TRUE
export TRAIN_BATCH_SIZE=4
export SAMPLE_BATCH_SIZE=4
export GRADIENT_CHECKPOINTING=FALSE
export GRADIENT_ACCUMULATION_STEPS=1
export SCALE_LR=FALSE
export LR_WARMUP_STEPS=0
export LR_NUM_CYCLES=1
export LR_POWER=1.0
export DATALOADER_NUM_WORKERS=4
export USE_8BIT_ADAM=TRUE
export MAX_GRAD_NORM=1.0
export PRIOR_GENERATION_PRECISION="fp32"

# default settings
export WORK_DIR="$HOME/JOBS/$JOB_ID"
export DATA_DIR="$WORK_DIR/data"
export REG_DATA_DIR="$DATA_DIR/reg"
export OUTPUT_DIR="$WORK_DIR/output"
export CHECKPOINT_PATH="$OUTPUT_DIR/{$MODEL}_$JOB_ID.safetensors"
export SCRIPT_PATH="$HOME/diffusers/examples/dreambooth/train_dreambooth_lora.py"
export CONVERT_SCRIPT_PATH="$HOME/scripts/convert_diffusers_to_original_stable_diffusion.py"

# set BASE_MODEL according to MODEL
# options:
#   - chilloutmix
#   - chilloutmix_100ep
#   - stable-diffusion-1.5
#   - stable-diffusion-2.1-base
#   - stable-diffusion-2.1
if [ "$MODEL" = "chilloutmix" ]; then
  export BASE_MODEL="$HOME/MODELS/chilloutmix"
elif [ "$MODEL" = "stable-diffusion-1.5" ]; then
  export BASE_MODEL="$HOME/MODELS/stable-diffusion-1.5"
elif [ "$MODEL" = "chilloutmix_100ep"]; then
	export BASE_MODEL="$HOME/MODELS/chilloutmix_100ep"
elif [ "$MODEL" = "stable-diffusion-2.1-base" ]; then
	export BASE_MODEL="$HOME/MODELS/stable-diffusion-2.1-base"
elif [ "$MODEL" = "stable-diffusion-2.1" ]; then
	export BASE_MODEL="$HOME/MODELS/stable-diffusion-2.1"
else
	echo "MODEL not supported"
	exit 1
fi

# create dirs
mkidr -p "$WORK_DIR"
mkidr -p "$DATA_DIR"
mkidr -p "$REG_DATA_DIR"
mkidr -p "$OUTPUT_DIR"

# download data and unzip it
gsutil cp "$DATA_REMOTE_PATH" "/tmp/$JOB_ID.zip"
unzip "/tmp/$JOB_ID.zip" -d "/tmp/$JOB_ID"
# cp the train dir to data dir, if not exist, exit
if [ ! -d "/tmp/$JOB_ID/train" ]; then
  echo "train dir not exist"
  exit 1
else
	# cp all images (.png .jpg .jpeg) under train dir to data dir
  find "/tmp/$JOB_ID/train" -type f -name "*.png" -exec cp {} "$DATA_DIR" \;
	find "/tmp/$JOB_ID/train" -type f -name "*.jpg" -exec cp {} "$DATA_DIR" \;
	find "/tmp/$JOB_ID/train" -type f -name "*.jpeg" -exec cp {} "$DATA_DIR" \;
fi
# cp the reg dir to reg data dir, if not exist, warn
if [ ! -d "/tmp/$JOB_ID/reg" ]; then
  echo "reg dir not exist"
else
  # cp all images (.png .jpg .jpeg) under reg dir to reg data dir
	find "/tmp/$JOB_ID/reg" -type f -name "*.png" -exec cp {} "$REG_DATA_DIR" \;
	find "/tmp/$JOB_ID/reg" -type f -name "*.jpg" -exec cp {} "$REG_DATA_DIR" \;
	find "/tmp/$JOB_ID/reg" -type f -name "*.jpeg" -exec cp {} "$REG_DATA_DIR" \;
fi

# delete the tmp data
rm -rf "/tmp/$JOB_ID.zip"
rm -rf "/tmp/$JOB_ID"

# update diffusers repo
cd "$HOME/diffusers" || exit
git pull

# go to the work dir
cd "$WORK_DIR" || exit
# active the env
source "$HOME/diffusers/.env/bin/activate"

# construct command
COMMAND = "accelerate launch $SCRIPT_PATH "
COMMAND += "--pretrained_model_name_or_path=$BASE_MODEL "
# --instance_data_dir
COMMAND += "--instance_data_dir=$DATA_DIR "
# if $REG_DATA_DIR is not empty, add --reg_data_dir
if [ "$(ls -A $REG_DATA_DIR)" ]; then
	COMMAND += "--reg_data_dir=$REG_DATA_DIR "
fi
# --instance_prompt
COMMAND += "--instance_prompt=$INSTANCE_PROMPT "
# --class_prompt
COMMAND += "--class_prompt=$CLASS_PROMPT "
# if $WITH_PRIOR_PRESERVATION is TRUE, add --with_prior_preservation
# also add --prior_loss_weight
if [ "$WITH_PRIOR_PRESERVATION" = "TRUE" ]; then
	COMMAND += "--with_prior_preservation "
	COMMAND += "--prior_loss_weight=$PRIOR_LOSS_WEIGHT "
fi
# --num_class_images
COMMAND += "--num_class_images=$NUM_CLASS_IMAGES "
# --output_dir
COMMAND += "--output_dir=$OUTPUT_DIR "
# --resolution
COMMAND += "--resolution=$RESOLUTION "
# if $CENTER_CROP is TRUE, add --center_crop
if [ "$CENTER_CROP" = "TRUE" ]; then
	COMMAND += "--center_crop "
fi
# if $TRAIN_TEXT_ENCODER is TRUE, add --train_text_encoder
if [ "$TRAIN_TEXT_ENCODER" = "TRUE" ]; then
	COMMAND += "--train_text_encoder "
fi
# --train_batch_size
COMMAND += "--train_batch_size=$TRAIN_BATCH_SIZE "
# --sample_batch_size
COMMAND += "--sample_batch_size=$SAMPLE_BATCH_SIZE "
# --num_train_epochs
COMMAND += "--num_train_epochs=$NUM_TRAIN_EPOCHS "
# --gradient_accumulation_steps
COMMAND += "--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
# if $GRADIENT_CHECKPOINTING is TRUE, add --gradient_checkpointing
if [ "$GRADIENT_CHECKPOINTING" = "TRUE" ]; then
	COMMAND += "--gradient_checkpointing "
fi
# --learning_rate
COMMAND += "--learning_rate=$LEARNING_RATE "
# if $SCALE_LR is TRUE, add --scale_lr
if [ "$SCALE_LR" = "TRUE" ]; then
	COMMAND += "--scale_lr "
fi
# --lr_scheduler
COMMAND += "--lr_scheduler=$LR_SCHEDULER "
# --lr_warmup_steps
COMMAND += "--lr_warmup_steps=$LR_WARMUP_STEPS "
# --lr_num_cycles
COMMAND += "--lr_num_cycles=$LR_NUM_CYCLES "
# --lr_power
COMMAND += "--lr_power=$LR_POWER "
# --dataloader_num_workers
COMMAND += "--dataloader_num_workers=$DATALOADER_NUM_WORKERS "
# if $USE_8BIT_ADAM is TRUE, add --use_8bit_adam
if [ "$USE_8BIT_ADAM" = "TRUE" ]; then
	COMMAND += "--use_8bit_adam "
fi
# --max_grad_norm
COMMAND += "--max_grad_norm=$MAX_GRAD_NORM "
# --mixed_precision
COMMAND += "--mixed_precision=$MIXED_PRECISION "
# --prior_generation_precision
COMMAND += "--prior_generation_precision=$PRIOR_GENERATION_PRECISION "
# if $ENABLE_XFORMERS is TRUE, add --enable_xformers_memory_efficient_attention
if [ "$ENABLE_XFORMERS" = "TRUE" ]; then
	COMMAND += "--enable_xformers_memory_efficient_attention "
fi

# echo the command
echo "$COMMAND"

# run the command
$COMMAND

# convert the output to safetensors
python3 $CONVERT_SCRIPT_PATH \
	--model_path="$OUTPUT_DIR" \
	--checkpoint_path="$CHECKPOINT_PATH" \
	--use_safetensors









