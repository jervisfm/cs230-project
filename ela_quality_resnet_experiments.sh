#!/bin/sh

set -x

NUM_ITER=15
REG=0
CUDA=True


# # Resnet
echo ">>> Training Resnet ela 85..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela85 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela85"
echo "Resnet Done"

echo ">>> Training Resnet ela 95..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela95 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela95"
echo "Resnet Done"

# # Repeat with full weights unlocked.

# Resnet
echo ">>> Training Resnet all weights ela85..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela85 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela85"
echo "Resnet Done"

echo ">>> Training Resnet all weights ela95..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela95 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela95"
echo "Resnet Done"


