
"""
Error analysis for the test set.
This is looking models that used ELA  (Error Level analysis) features
"""
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


parser.add_argument('--random', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Whether to do random picking of errors each time.")



FLAGS = parser.parse_args()

Model = collections.namedtuple('Model', 'name filepath datafolder')

models = [
Model('Alexnet', 'results/cnn_checkpoint_alexnet_l2reg=0_iter=100_ela.h5', 'data/processed_casia2_224_ela'),
Model('Resnet', 'results/cnn_checkpoint_resnet_pretrained_l2reg=0.005_iter=20_batchSize=100_learningRate=0.00025_trainallweights_raj.h5', 'data/processed_casia2_224_ela'), # This is actually an ela model.
Model('Densenet', 'results/cnn_checkpoint_densenet_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_224_ela'),
Model('Inception', 'results/cnn_checkpoint_inception_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_299_ela'),
Model('Vgg16', 'results/cnn_checkpoint_vgg16_pretrained_l2reg=0_iter=15_ela.h5', 'data/processed_casia2_224_ela'),
]


def get_model(modelname):
    """ Returns a model with the given name if one exists. """
    for model in models:
        if model.name.lower() == modelname:
            return model
    return None

def main():



    model = get_model("resnet")

    scores, labels, predicted_labels, loader = util.get_predicted_probs(model, FLAGS.cuda, return_dataloader=True)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, scores)
    f1_score = metrics.f1_score(labels, predicted_labels, pos_label=1)
    accuracy = metrics.accuracy_score(labels, predicted_labels)

    result = ''
    result += "\n\nDoing FINAL test results Error Analysis for Model: {}\n".format(model.name)
    result += "F1 score: {}\n".format(f1_score)
    result += "Accuracy: {}\n".format(accuracy)
    result += "AUC: {}\n".format(auc)


    # Find the indicies of the cases that we made mistakes and when we classified correctly.
    mistaken_indices = []
    correct_indicies = []
    for index, label in enumerate(labels):
        if labels[index] == predicted_labels[index]:
            # Correct prediction
            correct_indicies.append(index)
        else:
            # We made a mistake.
            mistaken_indices.append(index)


    # Sample out 10 random mistakes and 10 correct predictions.
    num_samples = 10
    num_correct_predictions = len(correct_indicies)
    num_mistaken_predictions = len(mistaken_indices)

    # Fix the randomness for reproducibility unless explicitly not requested.
    if not FLAGS.random:
        np.random.seed(42)
    result += '\n Number of error predictions: {}'.format(len(mistaken_indices))
    result += '\n Number of correct predictions: {}'.format(len(correct_indicies))
    result += '\n Accuracy sanity check for error analysis:  {}'.format( float(len(correct_indicies)) / (len(mistaken_indices) + len(correct_indicies)))

    sampled_mistaken_prediction_indices = np.random.choice(num_mistaken_predictions, num_samples)
    sampled_correct_prediction_indices = np.random.choice(num_correct_predictions, num_samples)

    result += '\n Sampled Mistaken prediction indices: {}'.format(sampled_mistaken_prediction_indices)
    result += '\n Sampled Correct prediction indices: {}'.format(sampled_correct_prediction_indices)



    result += '\n Mistaken Classifications: '
    for mistakened_index in sampled_mistaken_prediction_indices:
        filename = loader.filenames[mistakened_index]
        result += '\n{}'.format(filename)

    result += '\n Correct classifications: '
    for correct_index in sampled_correct_prediction_indices:
        filename = loader.filenames[correct_index]
        result += '\n{}'.format(filename)

    util.write_contents_to_file('error_analysis_resnet_ela.txt', result)



if __name__ == '__main__':
    main()