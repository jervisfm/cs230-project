
import os
import time
import torch
import util
import torch.nn as nn
import data_loader
import torchvision

from models.v1 import CNNv1
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
import subprocess

from bayes_opt import BayesianOptimization

import shlex

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=10, help="Number of iterations to perform bayesian optimization. The more steps the more likely to find a good maximum you are. ", type=int)
parser.add_argument('--init_points', default=2, help="How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.", type=int)

parser.add_argument('--cuda', default="True", help="Whether to use cuda.")
parser.add_argument('--data_folder', default="data/processed_casia2", help="Data folder with preprocessed CASIA data into train/dev/test splits.")
parser.add_argument('--model_name', default="alexnet", help="Name of CNN model to tune. Must be name of one of models available under models directory. e.g. {simple_cnn_v1} or in cnn.py")

FLAGS = parser.parse_args()


def train_function(max_iter=15, batch_size=100, learning_rate=0.001, l2_regularization=0, cuda=FLAGS.cuda, unfreeze_all_weights="True", unfreeze_ratio=1.0, model_name=FLAGS.model_name, data_folder=FLAGS.data_folder):
    """Executes training with given parameters. Returns best dev accuracy score."""
    max_iter = int(max_iter)
    batch_size = int(batch_size)
    shell_command="python cnn.py " \
                  "--max_iter {} " \
                  "--batch_size {} " \
                  "--learning_rate {} " \
                  "--model_name={} " \
                  "--data_folder={} " \
                  "--cuda={} " \
                  "--l2_regularization={} " \
                  "--unfreeze_all_weights={} " \
                  "--experiment_name 'l2reg={}_iter={}_trainallweights={}_unfreezeratio={}'".format(
        max_iter,
        batch_size,
        learning_rate,
        model_name,
        data_folder,
        cuda,
        l2_regularization,
        unfreeze_all_weights,
        l2_regularization,
        max_iter,
        unfreeze_all_weights,
        unfreeze_ratio
    )
    shell_command_list = shlex.split(shell_command)
    print(shell_command_list)
    result = subprocess.run(shell_command_list)
    return result.returncode

    #return -learning_rate * learning_rate

def main():
    # Bounded region of parameter space to explore.
    pbounds = {
        'learning_rate': (0.0001, 0.01),
        'l2_regularization': (0.00001, 0.02),
    }
    optimizer = BayesianOptimization(
        f=train_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=FLAGS.init_points,
        n_iter=FLAGS.max_iter,
    )

    print("Best Parameters are", optimizer.max)

if __name__ == '__main__':
    main()
