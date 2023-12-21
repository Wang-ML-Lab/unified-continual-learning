from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loaders = []
        self.test_loaders = []
        self.i = 0
        self.args = args

        if not all((self.NAME, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        return: 
            the **current** training and test loaders,
            the **next** training and test loaders.
        """
        raise NotImplementedError

    def step(self):
        self.i += 1

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError