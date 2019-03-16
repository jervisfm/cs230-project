#!/bin/sh

set -x

NUM_ITER=10
REG=0
CUDA=True

# # Resnet
echo ">>> Training Resnet..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Done"

echo ">>> Training Resnet2..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet2_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Done"


echo ">>> Training Resnet3..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet3_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Done"

echo ">>> Training Resnet4..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet4_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Done"


echo ">>> Training Resnet5..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet5_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Done"


# ####

# # Repeat with full weights unlocked.
# NUM_ITER=3

echo ">>> Training Resnet all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Done"

echo ">>> Training Resnet2 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet2_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Done"

echo ">>> Training Resnet3 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet3_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Done"


echo ">>> Training Resnet4 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet4_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Done"


echo ">>> Training Resnet5 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet5_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Done"


