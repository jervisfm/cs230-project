#!/bin/sh

set -x

NUM_ITER=20
CUDA=True

# REG=0
# L_R=0.001



# for BATCH_SIZE in 50 100 125
# do
# 	for L_R in 0.00025 0.0005 0.00075 0.001
# 	do
# 		for REG in 0 0.01 0.001 0.005
# 		do
# 			# # Resnet
# 			echo ">>> Training Resnet..."
# 			python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --unfreeze_all_weights=True --data_folder=data/processed_casia2_224 --cuda=$CUDA --batch_size=$BATCH_SIZE --learning_rate=$L_R --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batchSize=${BATCH_SIZE}_learningRate=${L_R}_trainallweights"
# 			echo "Resnet Done"
# 		done
# 	done
# done


for BATCH_SIZE in 50 100 125
do
	for L_R in 0.00025 0.0005 0.00075 0.001
	do
		for REG in 0 0.01 0.001 0.005
		do
			# # Resnet
			echo ">>> Training Resnet..."
			python cnn.py --max_iter $NUM_ITER --model_name=resnet_pretrained --unfreeze_all_weights=True --data_folder=data/processed_casia2_224_ela95 --cuda=$CUDA --batch_size=$BATCH_SIZE --learning_rate=$L_R --l2_regularization=$REG --experiment_name "l2reg=${REG}_iter=${NUM_ITER}_batchSize=${BATCH_SIZE}_learningRate=${L_R}_trainallweights_raj"
			echo "Resnet Done"
		done
	done
done