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

from datasets.transforms.rotation import FixedRotation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path

from datasets.mnist_perm import MyMNIST


class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
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
        step = 180 // self.N_TASKS
        for i in range(self.N_TASKS):
            lo, hi = i * step, (i+1) * step
            transform = transforms.Compose((
                FixedRotation(seed=np.random.randint(9999), deg_min=lo, deg_max=hi), 
                transforms.ToTensor()))
            train_dataset = MyMNIST(os.path.join(base_path(),'MNIST'),
                        train=True, download=True, transform=transform)
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                            transform, RotatedMNIST.NAME, self.args.validation_perc)
            else:
                test_dataset = MyMNIST(os.path.join(base_path(),'MNIST'),
                                    train=False, download=True, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)

    @staticmethod
    def get_backbone():
        return MNISTMLP((1, 28, 28), RotatedMNIST.N_CLASSES_PER_TASK)

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
        return RotatedMNIST.get_batch_size()
