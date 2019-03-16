#!/bin/sh

set -x

NUM_ITER=10
REG=0
CUDA=True

# # densenet
echo ">>> Training densenet..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "densenet Done"

echo ">>> Training densenet2..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet2_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "densenet Done"


echo ">>> Training densenet3..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet3_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "densenet Done"

echo ">>> Training densenet4..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet4_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}"
echo "densenet Done"


# ####

# # Repeat with full weights unlocked.
# NUM_ITER=3

echo ">>> Training densenet all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "densenet Done"

echo ">>> Training densenet2 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet2_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "densenet Done"

echo ">>> Training densenet3 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet3_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "densenet Done"


echo ">>> Training densenet4 all weights..."
python cnn.py --max_iter $NUM_ITER --model_name=densenet4_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --l2_regularization=$REG --unfreeze_all_weights=True --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_trainallweights"
echo "densenet Done"



