
import torch.nn as nn

class CNNv1(nn.Module):
    """ Initial version of our custom cnn architecture. """

    def __init__(self, input_size, num_classes):
        super(CNNv1, self).__init__()
        print("Input size is {}. Num Classes is {}".format(input_size, num_classes))


        # CNN output formula is :
        # floor of ((n + 2p - f) / stride) +  1.
        # We could use this to set the value. However, it is simpler to just configure
        # our cnn layer parameters, run what mismatches we get at runtime and update
        # final outputs accordingly.
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1, ),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2), )
        # Intermediate image is ~125x125

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2), )
        # Intermeidate image is 122x122

        self.output = nn.Sequential(nn.Linear(32 * 30 * 30, num_classes), nn.ReLU(),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten x
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
