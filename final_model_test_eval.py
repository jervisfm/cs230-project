
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

def get_predicted_probs(model):
    """Retruns predicted probs, actual_labels, predicted_labels for given model. """
    params = {'batch_size': 100, 'num_workers': 10, 'cuda': FLAGS.cuda}
    data_loaders = data_loader.fetch_dataloader(['test'], model.datafolder, params)
    loader = data_loaders['test']
    if FLAGS.cuda:
        torch_model = torch.load(model.filepath)
    else:
        torch_model = torch.load(model.filepath, map_location='cpu')
    scores = None
    actual_labels = None
    predicted_labels = []



    print ("Computing probabilities for model: ", model.name)
    num_batch = 0
    for images, labels in loader:
        if FLAGS.cuda:
            images, labels = images.cuda(async=True), labels.cuda(async=True)
        outputs = torch_model(images)
        print ("Processing batch #", num_batch)
        num_batch += 1
        if FLAGS.cuda:
            if model.name.lower().startswith("inception"):
                outputs = outputs[0].cuda()
            else:
                outputs = outputs.cuda()

        if scores is None:
            scores = outputs[:, 1]
        else:
            scores = torch.cat([scores.cpu(), outputs[:, 1].cpu()])
            scores = scores.detach().cpu()

        if actual_labels is None:
            actual_labels = labels
        else:
            actual_labels = torch.cat([actual_labels.cpu(), labels.cpu()])
            actual_labels = actual_labels.detach().cpu()

        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.append(predicted.cpu())

    print("Done. Scores shape: ", scores.shape)
    return scores.detach().cpu().numpy(), actual_labels.detach().cpu().numpy(), util.flatten_tensor_list(predicted_labels)


def main():
    results = ''
    for model in models:
        results += "Computing FINAL test results for Model: {}\n".format(model.name)
        scores, labels, predicted_labels = get_predicted_probs(model)
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