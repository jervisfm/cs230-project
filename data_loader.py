"""
PyTorch Based Data loader for the Casia2 dataset.

Based on https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py

"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import data_reader


DEFAULT_PARAMS = {
    'batch_size' : 100,
    'cuda' : 0,
    'num_workers': 10,
}
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


def fetch_dataloader(types, data_dir, params=None):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'dev', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) Optional hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    if params is None:
        print('Using Default params.')
        params = DEFAULT_PARAMS

    batch_size = params['batch_size']
    num_workers = params['num_workers']
    cuda = params['cuda']

    supported_partitions = set(['train', 'dev', 'test'])
    if not set(types).issubset(supported_partitions):
        raise ValueError("Unrecgonized split. Only support train/dev/test but got {}".format(types))

    for split in ['train', 'dev', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # Image transformer to get a torch tensor.
            transformer = transforms.Compose([
                #transforms.Resize(224),
                transforms.ToTensor()])


            if split == 'train':
                dataloader = DataLoader(Casia2Dataset(path, transformer), batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=cuda)
            else:
                dataloader = DataLoader(Casia2Dataset(path, transformer), batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=cuda)

            dataloaders[split] = dataloader


    return dataloaders