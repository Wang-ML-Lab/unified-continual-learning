# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import ipdb
import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms

from datasets import get_dataset
from torch.utils.data import DataLoader
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_backbone_args, ArgumentParser
from utils.status import progress_bar


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)

    def setup_joint_loader(self, dataset):
        trainsets = [x.dataset for x in dataset.train_loaders]
        comb_dataset = torch.utils.data.ConcatDataset(trainsets)
        # ipdb.set_trace()
        self.data_loader = DataLoader(comb_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.data_iter = iter(self.data_loader)

    def observe(self, cur_data, next_data):
        try: 
            x, y, idx = self.data_iter.next()
            x, y = x.to(self.device), y.to(self.device)
        except:
            self.data_iter = iter(self.data_loader)
            x, y, idx = self.data_iter.next()
            x, y = x.to(self.device), y.to(self.device)

        self.opt.zero_grad()
        outputs = self.net(x)
        loss = self.loss(outputs, y)
        loss.backward()
        self.opt.step()

        return loss.item()