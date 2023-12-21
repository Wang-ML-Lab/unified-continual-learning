# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Type
from argparse import Namespace
import os
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbones.mnistmlp import MNISTMLP
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from datasets.transforms.permutation import FixedPermutation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path



class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyMNIST, self).__init__(root, train, transform,
                                      target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target, index) where target is the id of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class PermutedMNIST(ContinualDataset):
    NAME = 'perm-mnist'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    INDIM = (1, 28, 28)
    MAX_N_SAMPLES_PER_TASK = 60000
    
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
        for _ in range(self.N_TASKS):
            transform = transforms.Compose((transforms.ToTensor(), FixedPermutation(seed=np.random.randint(9999))))
            train_dataset = MyMNIST(os.path.join(base_path(),'MNIST'),
                        train=True, download=True, transform=transform)
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                            transform, PermutedMNIST.NAME, self.args.validation_perc)
            else:
                test_dataset = MyMNIST(os.path.join(base_path(),'MNIST'),
                                    train=False, download=True, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, PermutedMNIST.N_CLASSES_PER_TASK)

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
        return PermutedMNIST.get_batch_size()
