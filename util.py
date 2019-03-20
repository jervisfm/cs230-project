"""
Based on https://github.com/jervisfm/cs229-project/blob/master/utils.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import data_loader

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def get_label(Y):
    labels = list(set(Y))
    output = []
    for element in Y:
        if element == 1:
            output.append('Fake')
        else:
            output.append('Real')
    return output


def create_confusion_matrices(Y_predicted, Y_actual, file_name):
    """
    Creates and save a confusion matrix with given filename.
    :param Y_predicted:
    :param Y_actual:
    :param file_name:
    :return:
    """
    np.set_printoptions(precision=2)

    # Generate confusion matrix
    class_names = ['Real', 'Fake']
    Y_actual = get_label(Y_actual)
    Y_predicted = get_label(Y_predicted)
    confusion = confusion_matrix(Y_actual, Y_predicted, labels=class_names)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(confusion, classes=class_names,
                          title='Confusion matrix, without normalization')
    fig1.savefig(file_name + '.png')
    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    fig2.savefig(file_name + '_norm.png')

def compute_precision_recall_f1_score(Y_predicted, Y_actual):
    """
    Returns a tuple (precision, recall, f1_score).
    :param Y_predicted:
    :param Y_actual:
    """
    class_names = ['Real', 'Fake']
    Y_actual = get_label(Y_actual)
    Y_predicted = get_label(Y_predicted)
    (precision, recall, f1score, _) = precision_recall_fscore_support(Y_actual, Y_predicted, labels=class_names, pos_label='Fake', average='binary')
    return (precision, recall, f1score)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




def get_predicted_probs(model, cuda, dataset='test', return_dataloader=False):
    """Retruns predicted probs, actual_labels, predicted_labels for given model. """
    params = {'batch_size': 100, 'num_workers': 10, 'cuda': cuda}
    data_loaders = data_loader.fetch_dataloader([dataset], model.datafolder, params)
    loader = data_loaders[dataset]
    if cuda:
        torch_model = torch.load(model.filepath)
    else:
        torch_model = torch.load(model.filepath, map_location='cpu')
    scores = None
    actual_labels = None
    predicted_labels = []



    print ("Computing probabilities for model: ", model.name)
    num_batch = 0
    for images, labels in loader:
        if cuda:
            images, labels = images.cuda(async=True), labels.cuda(async=True)
        outputs = torch_model(images)
        print ("Processing batch #", num_batch)
        num_batch += 1
        if cuda:
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
    if return_dataloader:
        return scores.detach().cpu().numpy(), actual_labels.detach().cpu().numpy(), flatten_tensor_list(
            predicted_labels), loader
    else:
        return scores.detach().cpu().numpy(), actual_labels.detach().cpu().numpy(), flatten_tensor_list(predicted_labels)



def flatten_tensor_list(prediction_list):
    output = []
    for tensor in prediction_list:
        for x in tensor:
            output.append(x)
    return output

def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)