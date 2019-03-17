#!/bin/sh

set -x

NUM_ITER=15
REG=0
CUDA=True


# # VGG
# echo ">>> Training VGG16 ..."
# python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224_ela  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela"
# echo "VGG Done"

# # Inception
# echo ">>> Training Inception..."
# python cnn.py --max_iter $NUM_ITER --model_name=inception_pretrained --data_folder=data/processed_casia2_299_ela  --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela"
# echo "Inception Done"

# # DenseNet
# echo ">>> Training Densenet..."
# python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela"
# echo "DenseNet Done"

# # Resnet
# echo ">>> Training Resnet..."
# python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_ela"
# echo "Resnet Done"

# # Alexnet Raw
# echo ">>> Training Alexnet Raw..."
# python cnn.py --max_iter 100 --model_name=alexnet --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=100_ela"
# echo "Alexnet Done"

# ####

# # Repeat with full weights unlocked.

# # Inception
# echo ">>> Training Inception all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=inception_pretrained --data_folder=data/processed_casia2_299_ela  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela"
# echo "Inception Done"

# # DenseNet
# echo ">>> Training Densenet all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela"
# echo "DenseNet Done"

# # Resnet
# echo ">>> Training Resnet all weights..."
# python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela"
# echo "Resnet Done"

# echo ">>> Training VGG16 all weights ..."
# python cnn.py --max_iter $NUM_ITER --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224_ela  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --batch_size=50 --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights_ela"
# echo "VGG Done"



REG=0.001
# Run Resnet with different params
echo ">>> Training Resnet all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --batch_size=50 --learning_rate=0.00025 --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batch_size=50_learning_rate=0.00025_trainallweights_ela"
echo "Resnet Done"

REG=0

# Re-run experiments which OOMd, with smaller batch size. 
#Inception
echo ">>> Training Inception all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=inception_pretrained --data_folder=data/processed_casia2_299_ela  --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --batch_size=50 --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batch_size=50_trainallweights_ela"
echo "Inception Done"


# DenseNet
echo ">>> Training Densenet all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --batch_size=50 --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batch_size=50_trainallweights_ela"
echo "DenseNet Done"

