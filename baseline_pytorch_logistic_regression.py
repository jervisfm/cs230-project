import torch
import torch.nn as nn
import data_loader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', default=100, help="Number of iterations to perform training.", type=int)
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
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


def train():
    params = {'batch_size': 100, 'num_workers': 100, 'cuda': 0}
    data_loaders = data_loader.fetch_dataloader(['train', 'dev'], FLAGS.data_folder)
    train_loader = data_loaders['train']
    dev_loader = data_loaders['dev']


    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(FLAGS.max_iter):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, FLAGS.max_iter, i + 1, len(train_loader) // batch_size, loss.data[0]))

    # Test the Model on dev data
    correct = 0
    total = 0
    for images, labels in dev_loader:
        images = Variable(images.view(-1, image_size * image_size))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    train()