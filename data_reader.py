import os

from scipy import ndimage


class dataReader:
    def __init__(self, folder):
        """
        Creates a new instance of data reader for the casia2 dataset.

        :param folder: Folder to read
        """
        self.folder = folder


        # Load Training Data.
        path = os.path.join(self.folder, 'train')
        for index, filename in enumerate(os.listdir(path)):
            if not filename.endswith('.jpg'):
                continue

            fullpath = path + filename
            print("Processing Training file #: {} - Name={}".format(index, filename))
            image_array = ndimage.imread(fullpath)
            print("Image array shape", image_array.shape)


if __name__ == '__main__':
    dataReader()