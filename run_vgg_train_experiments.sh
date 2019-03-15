#!/bin/sh

set -x

NUM_ITER=10
REG=0
CUDA=True


# # VGG
# echo ">>> Training VGG16 ..."
# python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
# echo "VGG Done"

#####

## Repeat with full weights unlocked.
NUM_ITER=3

RATIO=0.75
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO} --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

RATIO=0.5
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO} --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

RATIO=0.3
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO}  --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

RATIO=0.2
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO} --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

RATIO=0.1
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO} --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

RATIO=0.05
echo ">>> Training VGG16 all weights RATIO=${RATIO}..."
python cnn.py --unfreeze_ratio=${RATIO}  --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_unfreezeratio=${RATIO}_trainallweights"
echo "VGG Done"

####
exit 0

# Run the VGG model with L2 Regularization.
NUM_ITER=5
REG=0.01
echo ">>> Training VGG16 with regularization ..."
python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "VGG Done"

# Run the VGG model with L2 Regularization.
NUM_ITER=5
REG=0.02
echo ">>> Training VGG16 with regularization ..."
python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "VGG Done"

# Run the VGG model with L2 Regularization.
NUM_ITER=5
REG=0.03
echo ">>> Training VGG16 with regularization ..."
python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "VGG Done"


# Run the VGG model with L2 Regularization.
NUM_ITER=5
REG=0.04
echo ">>> Training VGG16 with regularization ..."
python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "VGG Done"

# Run the VGG model with L2 Regularization.
NUM_ITER=5
REG=0.05
echo ">>> Training VGG16 with regularization ..."
python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "VGG Done"

