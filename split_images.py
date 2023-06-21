import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import argparse

counter = 0
counter_lock = threading.Lock()

def resize_image(image, short_side=512):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)
    if height < width:
        new_height = short_side
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = short_side
        new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def central_crop(image, crop_width=512, crop_height=512):
    height, width = image.shape[:2]
    y = (height - crop_height) // 2
    x = (width - crop_width) // 2
    cropped_image = image[y:y + crop_height, x:x + crop_width]
    return cropped_image

def split_image(image, tile_size=512, overlap=0):
    height, width = image.shape[:2]
    tiles = []

    if max(height, width) <= 750:
        cropped_image = central_crop(image)
        tiles.append(cropped_image)
        return tiles

    for y in range(0, height - tile_size + 1, tile_size - overlap):
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

        if x + tile_size < width:
            tile = image[y:y + tile_size, width - tile_size:width]
            tiles.append(tile)

    if y + tile_size < height:
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile = image[height - tile_size:height, x:x + tile_size]
            tiles.append(tile)

        if x + tile_size < width:
            tile = image[height - tile_size:height, width - tile_size:width]
            tiles.append(tile)

    return tiles

def process_image(image_path, input_folder, output_folder):
    global counter
    global counter_lock

    image = cv2.imread(os.path.join(input_folder, image_path))
    resized_image = resize_image(image)
    tiles = split_image(resized_image)

    for tile in tiles:
        with counter_lock:
            file_number = f"{counter:08d}"
            counter += 1
        cv2.imwrite(os.path.join(output_folder, f"{file_number}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])

def main(input_folder, output_folder, num_threads=4):
    image_paths = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    process_image_partial = partial(process_image, input_folder=input_folder, output_folder=output_folder)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_image_partial, image_paths), total=len(image_paths)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing script.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where processed images will be saved.")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)