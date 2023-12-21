from typing import Tuple, Type
from argparse import Namespace
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbones.mnistmlp import MNISTMLP
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from datasets.transforms.permutation import FixedPermutation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path

base_path = '/data/local/core50/core50_128x128'
if not os.path.exists(base_path):
    base_path = '/data/local/haizhou/core50/core50_128x128'

import ipdb


class Core50(ImageFolder):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, transform=None, target_transform=None) -> None:
        super(Core50, self).__init__(root, transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target, index) where target is the id of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SequentialCore50(ContinualDataset):
    NAME = 'seq-core50'
    N_CLASSES_PER_TASK = 50
    N_TASKS = 11
    INDIM = (3, 128, 128)
    MAX_N_SAMPLES_PER_TASK = 16000
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.setup_loaders()

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i+1 < self.N_TASKS:
            next_train = self.train_loaders[self.i+1]
            next_test = self.test_loaders[self.i+1]
        
        return current_train, current_test, next_train, next_test

    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        for i in range(self.N_TASKS):
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            trainset_full = Core50(os.path.join(base_path, f's{i+1}'), transform=train_transform)
            testset_full = Core50(os.path.join(base_path, f's{i+1}'), transform=test_transform)

            length = len(trainset_full)
            train_length = int(0.8 * length)
            test_length = length - train_length
            train_dataset, _ = random_split(trainset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))
            _, test_dataset = random_split(testset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        return MNISTMLP(3 * 128 * 128, SequentialCore50.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return SequentialCore50.get_batch_size()

class TransferCore50(ContinualDataset):
    """
    Use 8 domains as training, remaining 3 domains (domain 3, 7, 10) as test.
    """
    NAME = 'trans-core50'
    N_CLASSES_PER_TASK = 50
    N_TASKS = 8
    INDIM = (3, 128, 128)
    MAX_N_SAMPLES_PER_TASK = 16000
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.setup_loaders()

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i+1 < self.N_TASKS:
            next_train = self.train_loaders[self.i+1]
            next_test = self.test_loaders[self.i+1]
        
        return current_train, current_test, next_train, next_test

    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        for i in range(self.N_TASKS):
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            trainset_full = Core50(os.path.join(base_path, f's{i+1}'), transform=train_transform)
            testset_full = Core50(os.path.join(base_path, f's{i+1}'), transform=test_transform)

            length = len(trainset_full)
            train_length = int(0.8 * length)
            test_length = length - train_length
            train_dataset, _ = random_split(trainset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))
            _, test_dataset = random_split(testset_full, [train_length, test_length], generator=torch.Generator().manual_seed(3407))

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        return MNISTMLP(3 * 128 * 128, SequentialCore50.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return SequentialCore50.get_batch_size()

if __name__ == '__main__':
    # args = Namespace()
    # args.batch_size = 128
    # args.num_workers = 16
    # dataset = SequentialCore50(args=args)
    dataset = Core50(root='/data/local/core50/core50_128x128/s1')
    ipdb.set_trace()