# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/caption.py

import os
import argparse
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import csv
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument("image_folder", type=str, help="Path to the input image folder")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16
)

model.to(device)

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return generated_text

output_file = "metadata.csv"

# Get a list of image files in the folder
image_files = [
    f
    for f in os.listdir(args.image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Write the results to the CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["filename", "text"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process image files sequentially and show progress with tqdm
    for filename in tqdm(image_files):
        image_path = os.path.join(args.image_folder, filename)
        caption = generate_caption(image_path)
        writer.writerow({"filename": filename, "text": caption})