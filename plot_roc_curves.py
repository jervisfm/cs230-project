
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


FLAGS = parser.parse_args()


Model = collections.namedtuple('Model', 'name filepath datafolder')


models = [
    Model('Resnet', 'results/cnn_checkpoint_resnet_pretrained_l2reg=0_iter=50.h5', 'data/processed_casia2_224'),

]

def get_predicted_probs(model):
    """Retruns predicted probs, labels for given model. """
    params = {'batch_size': 100, 'num_workers': 10, 'cuda': FLAGS.cuda}
    data_loaders = data_loader.fetch_dataloader(['dev'], model.datafolder, params)
    dev_loader = data_loaders['dev']
    if FLAGS.cuda:
        torch_model = torch.load(model.filepath)
    else:
        torch_model = torch.load(model.filepath, map_location='cpu')
    scores = None
    actual_labels = None
    print ("Computing probabilities for model: ", model.name)
    num_batch = 0
    for images, labels in dev_loader:
        if FLAGS.cuda:
            images, labels = images.cuda(async=True), labels.cuda(async=True)
        outputs = torch_model(images)
        print ("Processing batch #", num_batch)
        num_batch += 1
        if FLAGS.cuda:
            if FLAGS.model_name.lower().startswith("inception"):
                outputs = outputs[0].cuda()
            else:
                outputs = outputs.cuda()

        if scores is None:
            scores = outputs[:, 1]
        else:
            scores = torch.cat([scores, outputs[:, 1]])

        if actual_labels is None:
            actual_labels = labels
        else:
            actual_labels = torch.cat([actual_labels, labels])

    print("Done. Scores shape: ", scores.shape)
    return scores.detach().numpy(), actual_labels.detach().numpy()


def main():
    for model in models:
        scores, labels = get_predicted_probs(model)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc = metrics.roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, label="{}, auc={}".format(model.name, auc))
    plt.legend(loc=4)
    plt.title("ROC Curve of various models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('roc_curve_graph.png')

if __name__ == '__main__':
    main()