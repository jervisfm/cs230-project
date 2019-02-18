"""
Based on https://github.com/jervisfm/cs229-project/blob/master/utils.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def get_label(Y):
    labels = list(set(Y))
    output = []
    for element in Y:
        if element == 1:
            output.append('Fake')
        else:
            output.append('Real')
    print(output)
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


def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)