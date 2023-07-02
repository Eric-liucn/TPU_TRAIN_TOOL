#!/bin/bash

# wget https://github.com/Eric-liucn/TPU_TRAIN_TOOL/raw/main/diffusers_scripts/test.sh && chmod +x test.sh && ./test.sh

# Settings
export JOB_ID="000001"
export MODEL="chilloutmix_100ep"
export DATA_REMOTE_PATH="gs://aiforsure_ai/zip_datasets/jjj.zip"
export INSTANCE_PROMPT="jjj 1girl"
export CLASS_PROMPT="1girl"
export NUM_TRAIN_EPOCHS=100
export LEARNING_RATE=1e-4
export LR_SCHEDULER="constant"
export MIXED_PRECISION="fp16"
export ENABLE_XFORMERS=TRUE

# Unfrequent settings
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

# Directories and paths
export WORK_DIR="$HOME/JOBS/$JOB_ID"
export DATA_DIR="$WORK_DIR/data"
export REG_DATA_DIR="$WORK/reg"
export OUTPUT_DIR="$WORK_DIR/output"
export CHECKPOINT_PATH="$OUTPUT_DIR/$JOB_ID.safetensors"
export REMOTE_CHECKPOINT_PATH="gs://aiforsure_ai/train_output/dreambooth_lora/"$MODEL"_"$LEARING_RATE"_"$NUM_TRAIN_EPOCHS"/"$JOB_ID".safetensors"
export SCRIPT_PATH="$HOME/diffusers/examples/dreambooth/train_dreambooth_lora.py"
export CONVERT_SCRIPT_PATH="$HOME//diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"

# Set BASE_MODEL according to MODEL
case "$MODEL" in
  "chilloutmix") BASE_MODEL="$HOME/MODELS/chilloutmix" ;;
  "stable-diffusion-1.5") BASE_MODEL="$HOME/MODELS/stable-diffusion-1.5" ;;
  "chilloutmix_100ep") BASE_MODEL="$HOME/MODELS/chilloutmix_100ep" ;;
  "stable-diffusion-2.1-base") BASE_MODEL="$HOME/MODELS/stable-diffusion-2.1-base" ;;
  "stable-diffusion-2.1") BASE_MODEL="$HOME/MODELS/stable-diffusion-2.1" ;;
  *) echo "MODEL not supported"; exit 1 ;;
esac

# Create dirs
mkdir -p "$WORK_DIR" "$DATA_DIR" "$REG_DATA_DIR" "$OUTPUT_DIR"

# Download data and unzip it
gsutil cp "$DATA_REMOTE_PATH" "/tmp/$JOB_ID.zip"
unzip "/tmp/$JOB_ID.zip" -d "/tmp/$JOB_ID"

# Copy train and reg dirs, if they exist
for img_type in png jpg jpeg; do
  if [ -d "/tmp/$JOB_ID/train" ]; then
    find "/tmp/$JOB_ID/train" -type f -name "*.$img_type" -exec cp {} "$DATA_DIR" \;
  else
    echo "train dir not exist"; exit 1
  fi

  if [ -d "/tmp/$JOB_ID/reg" ]; then
    find "/tmp/$JOB_ID/reg" -type f -name "*.$img_type" -exec cp {} "$REG_DATA_DIR" \;
  else
    echo "reg dir not exist"
  fi
done

# Delete the tmp data
rm -rf "/tmp/$JOB_ID.zip" "/tmp/$JOB_ID"

# Update diffusers repo
cd "$HOME/diffusers" || exit
git pull

# Go to the work dir
cd "$WORK_DIR" || exit

# Activate the env
source "$HOME/diffusers/.env/bin/activate"

# Construct command
COMMAND="accelerate launch $SCRIPT_PATH "
COMMAND+="--pretrained_model_name_or_path=$BASE_MODEL "
COMMAND+="--instance_data_dir=$DATA_DIR "
COMMAND+="--reg_data_dir=$REG_DATA_DIR "
COMMAND+="--instance_prompt=\"$INSTANCE_PROMPT\" "
COMMAND+="--class_prompt=\"$CLASS_PROMPT\" "
if [ "$WITH_PRIOR_PRESERVATION" = "TRUE" ]; then
    COMMAND+="--with_prior_preservation "
    COMMAND+="--prior_loss_weight=$PRIOR_LOSS_WEIGHT "
fi
COMMAND+="--num_class_images=$NUM_CLASS_IMAGES "
COMMAND+="--output_dir=$OUTPUT_DIR "
COMMAND+="--resolution=$RESOLUTION "
if [ "$CENTER_CROP" = "TRUE" ]; then
    COMMAND+="--center_crop "
fi
if [ "$TRAIN_TEXT_ENCODER" = "TRUE" ]; then
    COMMAND+="--train_text_encoder "
fi
COMMAND+="--train_batch_size=$TRAIN_BATCH_SIZE "
COMMAND+="--sample_batch_size=$SAMPLE_BATCH_SIZE "
COMMAND+="--num_train_epochs=$NUM_TRAIN_EPOCHS "
COMMAND+="--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
if [ "$GRADIENT_CHECKPOINTING" = "TRUE" ]; then
    COMMAND+="--gradient_checkpointing "
fi
COMMAND+="--learning_rate=$LEARNING_RATE "
if [ "$SCALE_LR" = "TRUE" ]; then
    COMMAND+="--scale_lr "
fi
COMMAND+="--lr_scheduler=$LR_SCHEDULER "
COMMAND+="--lr_warmup_steps=$LR_WARMUP_STEPS "
COMMAND+="--lr_num_cycles=$LR_NUM_CYCLES "
COMMAND+="--lr_power=$LR_POWER "
COMMAND+="--dataloader_num_workers=$DATALOADER_NUM_WORKERS "
if [ "$USE_8BIT_ADAM" = "TRUE" ]; then
    COMMAND+="--use_8bit_adam "
fi
COMMAND+="--max_grad_norm=$MAX_GRAD_NORM "
COMMAND+="--mixed_precision=$MIXED_PRECISION "
COMMAND+="--prior_generation_precision=$PRIOR_GENERATION_PRECISION "
if [ "$ENABLE_XFORMERS" = "TRUE" ]; then
    COMMAND+="--enable_xformers_memory_efficient_attention "
fi


# Run training
echo "$COMMAND"
eval "$COMMAND"

# Convert safetensors to original stable diffusion model
python $CONVERT_SCRIPT_PATH \
	--model_path="$OUTPUT_DIR" \
	--checkpoint_path="$CHECKPOINT_PATH" \
	--use_safetensors

# Deactivate the env
deactivate

# upload checkpoint to gcs
gsutil cp "$CHECKPOINT_PATH" "$REMOTE_CHECKPOINT_PATH"