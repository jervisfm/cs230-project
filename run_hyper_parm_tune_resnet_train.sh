#!/bin/sh

set -x

NUM_ITER=15
CUDA=True

# REG=0
# L_R=0.001


# BATCH_SIZE=100


for BATCH_SIZE in 33 66 99
do
	for L_R in 0.001 0.004 0.008
	do
		for REG in 0 0.03 0.07
		do
			# # Resnet
			echo ">>> Training Resnet..."
			python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --data_folder=data/processed_casia2_224 --cuda=$CUDA --batch_size=$BATCH_SIZE --learning_rate=$L_R --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batchSize=${BATCH_SIZE}_learningRate=${L_R}"
			echo "Resnet Done"
		done
	done
done