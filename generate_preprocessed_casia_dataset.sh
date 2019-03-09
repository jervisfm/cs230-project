#!/bin/sh

set -x

echo "Generating Dataset for 64x64 images"
python preprocess_casia_dataset.py --image_size=64 --output_dir=data/processed_casia2_64

echo "Generating Dataset for 128x128 images"
python preprocess_casia_dataset.py --image_size=128 --output_dir=data/processed_casia2_128

echo "Generating Dataset for 224x224 images"
python preprocess_casia_dataset.py --image_size=224 --output_dir=data/processed_casia2_224

echo "Generating Dataset for 229x229 images"
python preprocess_casia_dataset.py --image_size=229 --output_dir=data/processed_casia2_229

echo "Generating Dataset for 512x512 images"
python preprocess_casia_dataset.py --image_size=512 --output_dir=data/processed_casia2_512

echo "Generating Dataset for 1024x1024 images"
python preprocess_casia_dataset.py --image_size=1024 --output_dir=data/processed_casia2_1024
