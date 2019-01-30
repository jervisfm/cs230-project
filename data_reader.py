import os

from scipy import ndimage
import numpy as np
from tqdm import tqdm

AUTHENTIC_PREFIX = 'Au'
FAKE_PREFIX = 'Tp'

def is_fake_image_file(filename):
    return filename.startswith(FAKE_PREFIX)

def is_real_image_file(filename):
    return filename.startswith(AUTHENTIC_PREFIX)

def get_x_y_data(path):
    """ Reads X/Y data matrices from the given folder path. """
    X = None
    Y = None

    for index, filename in tqdm(enumerate(os.listdir(path))):
        if not filename.endswith('.jpg'):
            continue

        fullpath = os.path.join(path, filename)
        #print("Processing  file #: {} - Name={}".format(index, filename))
        image_array = ndimage.imread(fullpath, mode="RGB")
        image_vector = image_array.reshape(1, -1).T

        y_label = np.full((1, 1), 1 if is_fake_image_file(filename) else 0)
        if X is None:
            X = image_vector
        else:
            X = np.concatenate((X, image_vector), axis=1)

        if Y is None:
            Y = y_label
        else:
            Y = np.concatenate((Y, y_label), axis=1)
    return X, Y

class dataReader:
    def __init__(self, folder='data/processed_casia2'):
        """
        Creates a new instance of data reader for the casia2 dataset.
        Note: data loading automatically happens at creation.

        :param folder: Folder to read
        """
        self.folder = folder


        # Load Training Data.
        print("Loading training data into memory ...")
        train_path = os.path.join(self.folder, 'train')
        self.X_train, self.Y_train = get_x_y_data(train_path)
        print("Done")

        # Load Dev Data.
        print("Loading dev data into memory ...")
        dev_path = os.path.join(self.folder, 'dev')
        self.X_dev, self.Y_dev = get_x_y_data(dev_path)
        print("Done")

        # Load test data.
        print("Loading test data into memory ...")
        test_path = os.path.join(self.folder, 'test')
        self.X_test, self.Y_test = get_x_y_data(test_path)
        print("Data loading complete.")

    def getTrain(self):
        """ Returns a tuple x,y for the training dataset. """
        return (self.X_train, self.Y_train)

    def getTest(self):
        """ Returns a tuple x,y for the test dataset. """
        return (self.X_test, self.Y_test)

    def getDev(self):
        """ Returns a tuple of x,y for the dev dataset. """
        return (self.X_dev, self.Y_dev)




if __name__ == '__main__':
    dataReader()