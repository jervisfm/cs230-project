
"""
A script to compute resutls for the final trained models on the test set.

"""
import torch
import util
import torch.nn as nn
import data_loader
import torchvision

from sklearn.metrics import classification_report
from models.v1 import CNNv1
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
from sklearn import metrics

from data_reader import dataReader

import numpy as np

import matplotlib.pyplot as plt
import collections


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.", type=int)

parser.add_argument('--cuda', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Whether to use cuda (gpu).")


FLAGS = parser.parse_args()

Model = collections.namedtuple('Model', 'name filepath datafolder')

models = [
Model('Alexnet', 'results/cnn_checkpoint_alexnet_l2reg=0_iter=100_ela.h5', 'data/processed_casia2_224_ela'),
Model('Resnet', 'results/cnn_checkpoint_resnet_pretrained_l2reg=0.005_iter=20_batchSize=100_learningRate=0.00025_trainallweights_raj.h5', 'data/processed_casia2_224_ela'), # This is actually an ela model.
Model('Densenet', 'results/cnn_checkpoint_densenet_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_224_ela'),
Model('Inception', 'results/cnn_checkpoint_inception_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_299_ela'),
Model('Vgg16', 'results/cnn_checkpoint_vgg16_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_224_ela'),
]

def main():
    results = ''
    for model in models:
        results += "\n\nComputing FINAL test results for Model: {}\n".format(model.name)
        scores, labels, predicted_labels = util.get_predicted_probs(model, FLAGS.cuda)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc = metrics.roc_auc_score(labels, scores)
        f1_score = metrics.f1_score(labels, predicted_labels, pos_label=1)
        accuracy = metrics.accuracy_score(labels, predicted_labels)

        class_names = ['Real', 'Fake']


        results += "F1 score: {}\n".format(f1_score)
        results += "Accuracy: {}\n".format(accuracy)
        results += "AUC: {}\n".format(auc)

        classification_report_string = classification_report(labels, predicted_labels, target_names=class_names)
        results += "\n"
        results += classification_report_string
        results += "----------"
    print(results)

    util.write_contents_to_file("final_test_results.txt", results)


if __name__ == '__main__':
    main()