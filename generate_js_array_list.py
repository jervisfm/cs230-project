"""
Simple script to genereate JS array listing.
"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import data_reader


import util


data_dir = 'data/processed_casia2_1024/dev/'

if __name__ == '__main__':
    filenames = os.listdir(data_dir)
    fullpaths = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg')]
    result = 'const IMAGE_FOLDER = "{}"; let files = ['.format(data_dir)
    for image_name in filenames:
        result += '"{}", '.format(image_name)
    result += '] ;'

    util.write_contents_to_file('image_files.js', result)
