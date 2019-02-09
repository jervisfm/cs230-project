"""
PyTorch Based Data loader for the Casia2 dataset.

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py

"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import data_reader


class Casia2Dataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__
    for the casia2 real/fake images data.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [f for f in self.filenames if f.endswith('.jpg')]
        self.full_paths = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [1 if(data_reader.is_fake_image_file(filename)) else 0 for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.full_paths[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'dev', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # Image transformer to get a torch tensor.
            transformer = transforms.Compose([
                transforms.ToTensor()])

            if split == 'train':
                dataloader = DataLoader(Casia2Dataset(path, transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dataloader = DataLoader(Casia2Dataset(path, transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dataloader

    return dataloaders