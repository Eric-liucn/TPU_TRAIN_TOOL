import cv2
import numpy as np

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

def split_image(image, tile_size=512, overlap=0):
    height, width = image.shape[:2]
    tiles = []

    for y in range(0, height - tile_size + 1, tile_size - overlap):
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

        # Add the last tile in the row
        if x + tile_size < width:
            tile = image[y:y + tile_size, width - tile_size:width]
            tiles.append(tile)

    # Add the last row of tiles
    if y + tile_size < height:
        for x in range(0, width - tile_size + 1, tile_size - overlap):
            tile = image[height - tile_size:height, x:x + tile_size]
            tiles.append(tile)

        # Add the bottom-right tile
        if x + tile_size < width:
            tile = image[height - tile_size:height, width - tile_size:width]
            tiles.append(tile)

    return tiles

def main():
    image_path = "input.jpg"
    output_folder = "output"

    image = cv2.imread(image_path)
    resized_image = resize_image(image)
    tiles = split_image(resized_image)

    for i, tile in enumerate(tiles):
        cv2.imwrite(f"{output_folder}/tile_{i}.jpg", tile)

if __name__ == "__main__":
    main()