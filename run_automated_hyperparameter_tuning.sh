#!/bin/sh

set -x

echo "Starting Hyperparameter tuning ..."
CUDA=True

# Want to spend 2 hours doing random explorations @ 30 minutes per trial.
INIT_POINTS=2
# Resnet model 1 epoch takes ~2 minutes. So for 15 epochs max, that's 30 minutes runtime.
# For an overnight run, we can explore 24 samples.
MAX_ITER=6
DATA_FOLDER="data/processed_casia2_224"

MODEL_NAME="resnet_pretrained"

python hyperparameter_tuning.py --cuda=$CUDA --init_points=$INIT_POINTS --max_iter=$MAX_ITER --data_folder=$DATA_FOLDER --model_name=$MODEL_NAME

echo "Done"