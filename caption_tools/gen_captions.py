# wget https://github.com/Eric-liucn/TPU_TRAIN_TOOL/raw/main/caption_tools/gen_captions.py

from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, VisionEncoderDecoderModel, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os
import glob
from pathlib import Path

# input a image, output a caption string
def caption(processor, model, image, device, tokenizer=None) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# get all images in a directory(include sub directories) that do not have a caption txt file in the same directory, return a list of image paths
# the image can in .png, .jpg, .jpeg format
def get_images_need_caption(directory) -> list:
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    image_paths = [image_path for image_path in image_paths if not os.path.exists(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.')[0] + '.txt'))]
    return image_paths

# @param image_paths: a list of image paths
# @param model_str: a string that specifies the model to use
# generate caption for all images in image_paths and save the caption in a txt file with the same name as the image
def create_captions(processor, model, image_paths, tokenizer=None):
    for image_path in image_paths:
        image = Image.open(image_path)
        caption_str = caption(processor, model, image, tokenizer)
        caption_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.')[0] + '.txt')
        with open(caption_path, 'w') as f:
            f.write(caption_str)

# main function
# get the image path from args
# get the model_str from args
# set device if args specifies, if not dont set device
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_str', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
        if device not in ['cpu', 'cuda']:
            raise ValueError('device must be cpu or cuda')
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_str == 'git_large_coco':
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
        tokenizer = None
    elif args.model_str == 'git_large_textcaps':
        processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps")
        tokenizer = None
    elif args.model_str == 'blip_base':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        tokenizer = None
    elif args.model_str == 'blip_large':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        tokenizer = None
    elif args.model_str == 'blip2':
        processor=Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
        model=Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)
        tokenizer = None
    elif args.model_str == 'vitgpt':
        processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    else:
        raise ValueError('model_str not supported')
    
    model.to(device)
    
    image_paths = get_images_need_caption(args.data_path)
    create_captions(processor, model, image_paths, device, tokenizer)
    

