"""
A simple script for processing casia 2 dataset.

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/build_dataset.py
"""
import argparse
import random
import os

from PIL import Image
from tqdm import tqdm


SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/CASIA2', help="Directory with the casia2 dataset")
parser.add_argument('--image_size', default=SIZE, help="Rescaled image size.", type=int)
parser.add_argument('--output_dir', default='data/processed_casia2', help="Where to write the preprocessed data")


def resize_and_save(filepath, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filepath)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    filename = filepath.split('/')[-1]

    # file name is now like foo.bmp. Let's convert it to .jpg
    filename = "%s.jpg" % (filename.split('.')[0])

    image.save(os.path.join(output_dir, filename))



def is_image_file(filename):
    return filename.endswith('.jpg') or filename.endswith('.tif') or filename.endswith('.bmp')

random_seed = 42
if __name__ == '__main__':
    # Fix random seed for reproducibility
    random.seed(random_seed)

    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    real_images_dir = os.path.join(args.data_dir, 'Au')
    fake_images_dir = os.path.join(args.data_dir, 'Tp')

    # Get the filenames in each directory
    real_images_filenames = os.listdir(real_images_dir)
    real_images_filenames = [os.path.join(real_images_dir, f) for f in real_images_filenames if is_image_file(f)]

    fake_images_filenames = os.listdir(fake_images_dir)
    fake_images_filenames = [os.path.join(fake_images_dir, f) for f in fake_images_filenames if is_image_file(f)]

    # Split into train / test / dev for each of the image classes.
    # We'll use a split ratio of 80% train, 10% dev, 10% test.
    # To get the 3 pieces, we'll do the splits as follows:
    # - Split dataset into 80/20 train/eval to get train set.
    # - Split eval piece 50/50 to get dev/test sets which each makes 10% of the overall data.


    real_images_filenames.sort()
    fake_images_filenames.sort()

    random.shuffle(real_images_filenames)
    random.shuffle(fake_images_filenames)

    train_split_ratio = 0.8
    dev_split_ratio = 0.1
    real_images_train_split = int(train_split_ratio * len(real_images_filenames))
    real_images_dev_split = int((train_split_ratio + dev_split_ratio) * len(real_images_filenames))

    fake_images_train_split = int(train_split_ratio * len(fake_images_filenames))
    fake_images_dev_split = int((train_split_ratio + dev_split_ratio) * len(fake_images_filenames))

    print("Real images: train_index={} dev_index={}", real_images_train_split, real_images_dev_split)
    print("Fake images: train_index={} dev_index={}", fake_images_train_split, fake_images_dev_split)
    train_filenames = real_images_filenames[:real_images_train_split] + fake_images_filenames[:fake_images_train_split]

    dev_filenames = real_images_filenames[real_images_train_split:real_images_dev_split] + fake_images_filenames[fake_images_train_split:fake_images_dev_split]

    test_filenames = real_images_filenames[real_images_dev_split:] + fake_images_filenames[fake_images_dev_split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=args.image_size)

    print("Done building dataset")