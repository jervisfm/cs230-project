#!/bin/sh

set -x

echo "Starting Hyperparameter tuning ..."
CUDA=False
INIT_POINTS=1
MAX_ITER=1
DATA_FOLDER="data/processed_casia2_224"

MODEL_NAME="resnet_pretrained"

python hyperparameter_tuning.py --cuda=$CUDA --init_points=$INIT_POINTS --max_iter=$MAX_ITER --data_folder=$DATA_FOLDER --model_name=$MODEL_NAME

echo "Done"