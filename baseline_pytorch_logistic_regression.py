
import os
import time
import torch
import torch.nn as nn
import data_loader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.", type=int)
parser.add_argument('--batch_size', default=100, help="Number of examples in one batch of minigradient descent.", type=int)
parser.add_argument('--num_workers', default=20, help="Number of workers to use in loading data.", type=int)
parser.add_argument('--cuda', default=False, help="Wheter to use cuda (gpu) for training.", type=bool)
parser.add_argument('--data_folder', default="data/processed_casia2", help="Data folder with preprocessed CASIA data into train/dev/test splits.")
parser.add_argument('--results_folder', default='results/', help="Where to write any results.")
parser.add_argument('--experiment_name', default=None, help="Name for the experiment. Useful for tagging files.")

FLAGS = parser.parse_args()

# Hyper Parameters
image_size = 128
input_size = image_size**2 * 3
num_classes = 2
#num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        print("Input size is {}. Num Classes is {}".format(input_size, num_classes))
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("baseline_pytorch_logistic_regression_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_train_dev_error_graph_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("baseline_pytorch_logistic_regression_train_dev_error_per_epoch", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)

def append_to_file(output_file, data):
    with open(output_file, 'a') as file_handle:
        file_handle.write(data)
        file_handle.write('\n')

def eval_on_train_set(model, train_loader):
    correct = 0
    total = 0
    for images, labels in train_loader:
        if FLAGS.cuda:
          images, labels = images.cuda(async=True), labels.cuda(async=True)
        images = Variable(images.view(-1, input_size))
        outputs = model(images).cuda() if FLAGS.cuda else model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the training set of images: %d %%' % (accuracy))
    return accuracy


def eval_on_dev_set(model, dev_loader):
    correct = 0
    total = 0
    for images, labels in dev_loader:
        if FLAGS.cuda:
          images, labels = images.cuda(async=True), labels.cuda(async=True)

        images = Variable(images.view(-1, input_size))
        outputs = model(images).cuda() if FLAGS.cuda else model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the dev set of images: %d %%' % (accuracy))
    return accuracy

def train():
    params = {'batch_size': FLAGS.batch_size, 'num_workers': FLAGS.num_workers, 'cuda': FLAGS.cuda}
    data_loaders = data_loader.fetch_dataloader(['train', 'dev'], FLAGS.data_folder, params)
    train_loader = data_loaders['train']
    dev_loader = data_loaders['dev']

    model = LogisticRegression(input_size, num_classes).cuda() if FLAGS.cuda \
        else LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training the Model
    start_time_secs = time.time()
    train_dev_error_graph_filename = get_train_dev_error_graph_filename()
    append_to_file(train_dev_error_graph_filename, 'epoch,train_accuracy,dev_accuracy')
    for epoch in range(FLAGS.max_iter):
        for i, (images, labels) in enumerate(train_loader):

            if FLAGS.cuda:
                images, labels = images.cuda(async=True), labels.cuda(async=True)

            images = Variable(images.view(-1, input_size))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, FLAGS.max_iter, i + 1, len(train_loader) // batch_size, loss.item()))

        train_acc = eval_on_train_set(model, train_loader)
        dev_acc = eval_on_dev_set(model, dev_loader)
        append_to_file(train_dev_error_graph_filename, '%s,%s,%s' % (epoch, train_acc, dev_acc))


    print('Training Complete')
    end_time_secs = time.time()
    training_duration_secs = end_time_secs - start_time_secs

    # Test the Model on dev data
    print('Final Evaluations after TRAINING...')
    train_accuracy = eval_on_train_set(model, train_loader)
    # Test on the train model to see how we do on that as well.
    dev_accuracy = eval_on_dev_set(model, dev_loader)

    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nDev Acurracy: {}%".format(dev_accuracy)
    experiment_result_string += "\nTrain Acurracy: {}%".format(train_accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format(
        1.0 * training_duration_secs / FLAGS.max_iter)

    print(experiment_result_string)
    # Save report to file
    write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

if __name__ == '__main__':
    train()
