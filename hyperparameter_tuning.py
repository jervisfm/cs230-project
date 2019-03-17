
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

parser.add_argument('--data_folder', default="data/processed_casia2", help="Data folder with preprocessed CASIA data into train/dev/test splits.")
parser.add_argument('--model_name', default="alexnet", help="Name of CNN model to tune. Must be name of one of models available under models directory. e.g. {simple_cnn_v1} or in cnn.py")

FLAGS = parser.parse_args()


def train_function(max_iter=100, batch_size=100, learning_rate=0.001, l2_regularization=0, cuda=False, unfreeze_all_weights=False, unfreeze_ratio=1.0, model_name="v1", data_folder=""):
    """Executes training with given parameters. Returns best dev accuracy score."""
    #shell_command="python cnn.py --max_iter {} --model_name={} --data_folder={} --cuda={} --l2_regularization={} --unfreeze_all_weights={} --experiment_name 'l2reg={}_iter={}_trainallweights={}_unfreezeratio={}'".format ()
    shell_command="/Users/jmuindi/cs230/cs230_project/foo.sh {}".format(int(learning_rate))
    shell_command_list = shlex.split(shell_command)
    print(shell_command_list)
    result = subprocess.run(shell_command_list)
    return result.returncode

    #return -learning_rate * learning_rate

def main():
    # Bounded region of parameter space
    pbounds = {'learning_rate': (1, 10)}
    optimizer = BayesianOptimization(
        f=train_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    print("Best Parameters are", optimizer.max)

if __name__ == '__main__':
    main()
