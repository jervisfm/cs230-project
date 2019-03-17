
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.", type=int)
parser.add_argument('--save_model_every_num_epoch', default=10, help="Save/checkpoing model every given number of epochs while training.", type=int)
parser.add_argument('--batch_size', default=100, help="Number of examples in one batch of minigradient descent.", type=int)
parser.add_argument('--num_workers', default=20, help="Number of workers to use in loading data.", type=int)
parser.add_argument('--learning_rate', default=0.001, help="Learning Rate hyperparameter.", type=float)
parser.add_argument('--l2_regularization', default=0.0, help="Regularization parameter lambda for L2 regularization.", type=float)
parser.add_argument('--cuda', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Whether to use cuda (gpu) for training.")
parser.add_argument('--unfreeze_all_weights', default=False, help="When using a pretrained model, whether to unfreeze all weights and make them trainable as well.", type=bool)
parser.add_argument('--unfreeze_ratio', default=1.0, help="Ratio of weights to be unfrozen. 1.0 to have all weights unfrozen.", type=float)
parser.add_argument('--data_folder', default="data/processed_casia2", help="Data folder with preprocessed CASIA data into train/dev/test splits.")
parser.add_argument('--model_name', default="alexnet", help="Name of CNN model to train. Must be name of one of models available under models directory. e.g. {simple_cnn_v1}")
parser.add_argument('--results_folder', default='results/', help="Where to write any results.")
parser.add_argument('--experiment_name', default=None, help="Name for the experiment. Useful for tagging files.")

FLAGS = parser.parse_args()

# Hyper Parameters
image_size = 128
input_size = image_size**2 * 3
num_classes = 2
#num_epochs = 5
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate

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
    experiment_name = "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""
    model_name = "_" + FLAGS.model_name
    return "{}{}".format(model_name, experiment_name)

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("cnn_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_model_checkpoint_path():
    suffix_name = get_suffix_name()
    filename = "{}{}.h5".format("cnn_checkpoint", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_train_dev_error_graph_filename(write_file=True):
    suffix_name = get_suffix_name()
    filename = "{}{}.csv".format("cnn_train_dev_error_per_epoch", suffix_name)
    path =  os.path.join(FLAGS.results_folder, filename)

    if write_file:
        write_contents_to_file(path, 'epoch,train_accuracy,dev_accuracy\n')
    return path

def get_training_loss_graph_filename(write_file=True):
    suffix_name = get_suffix_name()
    filename = "{}{}.csv".format("cnn_train_loss_per_minibatch", suffix_name)
    path = os.path.join(FLAGS.results_folder, filename)
    if write_file:
        write_contents_to_file(path, 'mini_batch_iteration,loss\n')
    return path

def get_confusion_matrix_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format("cnn_confusion_matrix", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)

def append_to_file(output_file, data):
    with open(output_file, 'a') as file_handle:
        file_handle.write(data)
        file_handle.write('\n')

def flatten_tensor_list(prediction_list):
    output = []
    for tensor in prediction_list:
        for x in tensor:
            output.append(x)
    return output

def get_num_model_parameters(model_parameters):
    count = 0
    for index, param in model_parameters:
        count += 1
    return count

def eval_on_train_set(model, train_loader):
    correct = 0
    total = 0
    for images, labels in train_loader:
        if FLAGS.cuda:
          images, labels = images.cuda(async=True), labels.cuda(async=True)
        #images = Variable(images.view(-1, input_size))
        outputs = model(images)
        if FLAGS.cuda:
            if FLAGS.model_name.lower().startswith("inception"):
                outputs = outputs[0].cuda()
            else:
                outputs = outputs.cuda()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the training set of images: %d %%' % (accuracy))
    return accuracy


def eval_on_dev_set(model, dev_loader):
    correct = 0
    total = 0
    y_true = []
    y_predicted = []

    for images, labels in dev_loader:
        if FLAGS.cuda:
          images, labels = images.cuda(async=True), labels.cuda(async=True)

        #images = Variable(images.view(-1, input_size))
        outputs = model(images)
        if FLAGS.cuda:
            if FLAGS.model_name.lower().startswith("inception"):
                outputs = outputs[0].cuda()
            else:
                outputs = outputs.cuda()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        y_predicted.append(predicted.cpu())
        y_true.append(labels.cpu())

    accuracy = 100 * correct / total
    print('Accuracy of the model on the dev set of images: %d %%' % (accuracy))
    return accuracy, flatten_tensor_list(y_predicted), flatten_tensor_list(y_true)

def get_model():
    """ Returns the model to use for training. """
    model_name = FLAGS.model_name.lower()
    unfreeze_weights = FLAGS.unfreeze_all_weights
    unfreeze_ratio = FLAGS.unfreeze_ratio
    if model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=False, num_classes=num_classes)
    elif model_name == 'alexnet_pretrained':
        model = torchvision.models.alexnet(pretrained=True)
        for i, param in model.named_parameters():
            param.requires_grad = unfreeze_weights
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        for name, params in model.named_parameters():
          print(name, params.requires_grad)
    elif model_name == 'inception':
        model = torchvision.models.inception_v3(pretrained=False, num_classes=num_classes)
    elif model_name == 'inception_pretrained':
        model = torchvision.models.inception_v3(pretrained=True)
        for i, param in model.named_parameters():
            param.requires_grad = unfreeze_weights
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        for name, params in model.named_parameters():
          print(name, params.requires_grad)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False, num_classes=num_classes)
    elif model_name == 'vgg16_pretrained':
        model = torchvision.models.vgg16(pretrained=True)
        model_parameters = model.named_parameters()
        num_model_parameters = get_num_model_parameters(model_parameters)
        max_num_parameters = num_model_parameters * unfreeze_ratio
        print("Num Model traininable parameters: ", num_model_parameters)
        print("Limiting to max num training parameters", max_num_parameters)
        for index, (name, param) in enumerate(model.named_parameters()):
            if (unfreeze_weights and index >= max_num_parameters):
                param.requires_grad = False
                continue
            param.requires_grad = unfreeze_weights
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))
        for name, params in model.named_parameters():
            print(name, params.requires_grad)
    elif model_name in ['densenet', 'densenet2', 'densenet3', 'densenet4']:
        model_init_mapping = {'densenet' : torchvision.models.densenet121, 'densenet2': torchvision.models.densenet161, 'densenet3': torchvision.models.densenet169, 'densenet4': torchvision.models.densenet201}
        model = model_init_mapping[model_name](pretrained=False, num_classes=num_classes)
    elif model_name in ['densenet_pretrained', 'densenet2_pretrained', 'densenet3_pretrained', 'densenet4_pretrained']:
        model_init_mapping = {'densenet_pretrained': torchvision.models.densenet121, 'densenet2_pretrained': torchvision.models.densenet161,
                              'densenet3_pretrained': torchvision.models.densenet169, 'densenet4_pretrained': torchvision.models.densenet201}
        num_features_mapping = {'densenet_pretrained': 1024, # 1024
                              'densenet2_pretrained': 2208,
                              'densenet3_pretrained': 1664,
                              'densenet4_pretrained': 1920}
        model = model_init_mapping[model_name](pretrained=True)
        for i, param in model.named_parameters():
            param.requires_grad = unfreeze_weights
        num_features = num_features_mapping[model_name]
        model.classifier = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_features, num_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_features, num_classes))
        for name, params in model.named_parameters():
            print(name, params.requires_grad)
    elif model_name in ['resnet', 'resnet2', 'resnet3', 'resnet4', 'resnet5']:
        model_init_mapping = {'resnet': torchvision.models.resnet18, 'resnet2': torchvision.models.resnet34,
                              'resnet3': torchvision.models.resnet50, 'resnet4': torchvision.models.resnet101,
                              'resnet5': torchvision.models.resnet152}
        model = model_init_mapping[model_name](pretrained=False, num_classes=num_classes)
    elif model_name in ['resnet_pretrained', 'resnet2_pretrained', 'resnet3_pretrained', 'resnet4_pretrained', 'resnet5_pretrained']:
        model_init_mapping = {'resnet_pretrained': torchvision.models.resnet18,
                              'resnet2_pretrained': torchvision.models.resnet34,
                              'resnet3_pretrained': torchvision.models.resnet50,
                              'resnet4_pretrained': torchvision.models.resnet101,
                              'resnet5_pretrained': torchvision.models.resnet152}
        model = model_init_mapping[model_name](pretrained=True)
        num_features_mapping = {'resnet_pretrained': 512,
                              'resnet2_pretrained': 512,
                              'resnet3_pretrained': 2048,
                              'resnet4_pretrained': 2048,
                              'resnet5_pretrained': 2048}
        num_features = num_features_mapping[model_name]
        for i, param in model.named_parameters():
            param.requires_grad = unfreeze_weights
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_features, num_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_features, num_classes))
        for name, params in model.named_parameters():
            print(name, params.requires_grad)
    elif model_name == 'v1':
        model = CNNv1(input_size, num_classes)
    else:
        raise ValueError('Got unexpected model: ', FLAGS.model_name)

    model = model.cuda() if FLAGS.cuda else model
    return model

def train():
    params = {'batch_size': FLAGS.batch_size, 'num_workers': FLAGS.num_workers, 'cuda': FLAGS.cuda}
    data_loaders = data_loader.fetch_dataloader(['train', 'dev'], FLAGS.data_folder, params)
    train_loader = data_loaders['train']
    dev_loader = data_loaders['dev']
    learning_rate = FLAGS.learning_rate

    model = get_model()

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=FLAGS.l2_regularization)

    # Training the Model
    start_time_secs = time.time()
    train_dev_error_graph_filename = get_train_dev_error_graph_filename()
    train_loss_graph_filename = get_training_loss_graph_filename()
    num_iteration = 0

    print("Model arch: ", model)
    print("Model size is ", sum([param.nelement() for param in model.parameters()]))
    dev_accuracy_list = []
    for epoch in range(FLAGS.max_iter):
        for i, (images, labels) in enumerate(train_loader):

            if FLAGS.cuda:
                images, labels = images.cuda(async=True), labels.cuda(async=True)

            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            if FLAGS.model_name.lower().startswith("inception"):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            total_data_set_size = len(train_loader.dataset)
            num_steps = total_data_set_size // batch_size
            if (i + 1) % 10 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, FLAGS.max_iter, i + 1, num_steps, loss.item()))
            append_to_file(train_loss_graph_filename, "%d,%.4f" % (num_iteration, loss.item()))
            num_iteration += 1

        train_acc = eval_on_train_set(model, train_loader)
        dev_acc, y_dev_predicted, y_dev_true = eval_on_dev_set(model, dev_loader)
        dev_accuracy_list.append(dev_acc)
        append_to_file(train_dev_error_graph_filename, '%s,%s,%s' % (epoch, train_acc.item()/100, dev_acc.item()/100))

        if (epoch + 1) % FLAGS.save_model_every_num_epoch == 0:
            print('Checkpointing model...')
            torch.save(model, get_model_checkpoint_path())


    print('Training Complete')
    end_time_secs = time.time()
    training_duration_secs = end_time_secs - start_time_secs

    print('Checkpointing FINAL trained model...')
    torch.save(model, get_model_checkpoint_path())

    print('Final Evaluations after TRAINING...')
    # Test on the train model to see how we do on that as well.
    train_accuracy = eval_on_train_set(model, train_loader)
    # Test the Model on dev data
    dev_accuracy, y_dev_predicted, y_dev_true = eval_on_dev_set(model, dev_loader)
    dev_accuracy_list.append(dev_accuracy)
    best_dev_accuracy = max(dev_accuracy_list)
    best_dev_accuracy_index = dev_accuracy_list.index(best_dev_accuracy)
    best_dev_accuracy_epoch = best_dev_accuracy_index + 1

    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nDev Acurracy: {}%".format(dev_accuracy)
    experiment_result_string += "\nBest Dev Acurracy over training: {}% seen at epoch {}".format(best_dev_accuracy, best_dev_accuracy_epoch)
    experiment_result_string += "\nTrain Acurracy: {}%".format(train_accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format(
        1.0 * training_duration_secs / FLAGS.max_iter)

    print(experiment_result_string)
    # Save report to file
    write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    # Generate confusion matrix
    util.create_confusion_matrices(y_dev_predicted, y_dev_true, get_confusion_matrix_filename())

    return best_dev_accuracy

if __name__ == '__main__':
    best_dev_accuracy = train()
    # Note: We return the best dev accuracy in the exit code so that we can use this in our hyperparameter tuning
    # script to do more automated parameter tuning.
    exit(best_dev_accuracy.item())
