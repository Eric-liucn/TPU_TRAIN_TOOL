# my_script.py

import os
import argparse
from diffusers import StableDiffusionPipeline
from diffusers import FlaxStableDiffusionPipeline


def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error(f"The directory {arg} does not exist.")
    else:
        return arg


parser = argparse.ArgumentParser(description='Description of your script')

parser.add_argument("mode", type=str, choices=["fp", "pf"],
                    help="Description of mode")
parser.add_argument('input', type=lambda x: is_valid_directory(parser, x), help='Description of dir1')
parser.add_argument('output', type=lambda x: is_valid_directory(parser, x), help='Description of dir2')

args = parser.parse_args()


def convert_flax_to_pytorch(input_dir, output_dir):
    pipeline = StableDiffusionPipeline.from_pretrained(input_dir, from_flax=True, safety_checker=None)
    pipeline.save_pretrained(output_dir)


def convert_pytorch_to_flax(input_dir, output_dir):
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(input_dir, from_pt=True, safety_checker=False)
    pipeline.save_pretrained(output_dir, params=params)


if args.mode == "fp":
    convert_flax_to_pytorch(args.input, args.output)
elif args.mode == "pf":
    convert_pytorch_to_flax(args.input, args.output)
else:
    exit(1)
