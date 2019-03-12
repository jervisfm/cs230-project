#!/bin/sh

set -x

NUM_ITER=10
REG=0
CUDA=True


# # VGG
# echo ">>> Training VGG16 ..."
# python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
# echo "VGG Done"

# # Inception
# echo ">>> Training Inception..."
# python cnn.py --max_iter $NUM_ITER --model_name=inception_pretrained --data_folder=data/processed_casia2_299  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
# echo "Inception Done"

# # DenseNet
# echo ">>> Training Densenet..."
# python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
# echo "DenseNet Done"

# # Resnet
# echo ">>> Training Resnet..."
# python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
# echo "Resnet Done"

# # Alexnet Raw
# echo ">>> Training Alexnet Raw..."
# python cnn.py --max_iter 100 --model_name=alexnet --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=100"
# echo "Alexnet Done"

# ####

# # Repeat with full weights unlocked.
# NUM_ITER=3

# # Inception
# echo ">>> Training Inception all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=inception_pretrained --data_folder=data/processed_casia2_299  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
# echo "Inception Done"

# # DenseNet
# echo ">>> Training Densenet all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
# echo "DenseNet Done"

# # Resnet
# echo ">>> Training Resnet all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
# echo "Resnet Done"

# echo ">>> Training VGG16 all weights ..."
# python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
# echo "VGG Done"

####

# Run only promising network(s) for more iterations 
NUM_ITER=50

# Resnet pretrained
echo ">>> Training Pre-trained Resnet..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Pretrained Done"

# # Resnet train all weights
echo ">>> Training Resnet all layers..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Train all layers Done"


REG=0.1


# Resnet pretrained
echo ">>> Training Pre-trained Resnet..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Pretrained Done"

# # Resnet train all weights
echo ">>> Training Resnet all layers..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Train all layers Done"


REG=0.2

# Resnet pretrained
echo ">>> Training Pre-trained Resnet..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "Resnet Pretrained Done"

# # Resnet train all weights
echo ">>> Training Resnet all layers..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "Resnet Train all layers Done"



