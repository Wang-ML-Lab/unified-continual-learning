# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import suppress
import os
import sys
from typing import Any, Dict

import numpy as np

from utils import create_if_not_exists
from utils.conf import base_path
from utils.metrics import average_i, average_iplus1, backward_transfer, forward_transfer, forgetting

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class Logger:
    def __init__(self, dataset_str: str,
                 model_str: str) -> None:
        self.accs = [] # average accs 
        self.accs_iplus1 = [] # average accs
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.bwt = None
        self.forgetting = None
        self.setting = '.'.join([dataset_str, model_str])

    def dump(self):
        return {
            'acc_matrix': self.acc_matrix,
            'accs': self.accs,
            'accs_iplus1': self.accs_iplus1,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
        }

    def load(self, dic):
        self.acc_matrix = dic['acc_matrix']
        self.accs = dic['accs']
        self.accs_iplus1 = dic['accs_iplus1']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']

    def rewind(self, num):
        # not sure when this is useful.
        self.accs = self.accs[:-num]
        self.accs_iplus1 = self.accs_iplus1[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]

    def add_fwt(self, results, accs):
        """accs: random baseline accuracies."""
        self.fwt = forward_transfer(results, accs)
        return self.fwt

    def add_bwt(self, results):
        self.bwt = backward_transfer(results)
        return self.bwt

    def add_forgetting(self, results):
        self.forgetting = forgetting(results)
        return self.forgetting

    def add_average_i(self, results, i):
        acc = average_i(results, i)
        self.accs.append(acc)
        return acc

    def add_average_iplus1(self, results, i):
        acc = average_iplus1(results, i)
        self.accs_iplus1.append(acc)
        return acc

    def add_acc_matrix(self, results):
        self.acc_matrix = np.array(results)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        wrargs['acc_matrix'] = self.acc_matrix

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc
        for i, acc in enumerate(self.accs_iplus1):
            wrargs['accmean_iplus1_task' + str(i + 1)] = acc

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        target_folder = os.path.join(base_path(), "results")

        create_if_not_exists(target_folder + self.setting)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset + "/" + self.model)

        path = target_folder + self.setting + "/" + self.dataset\
            + "/" + self.model + "/logs.pyd"
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')
