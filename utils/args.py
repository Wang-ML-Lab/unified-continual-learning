# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from backbones import get_all_backbones
from utils import get_all_losses
from models.utils.continual_model import optimizer_dict


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
                            
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--loss', type=str, default='ce',
                        help='Loss name.', choices=get_all_losses())
    parser.add_argument('--opt', type=str, default='sgd',
                        help='Optimizer type.', choices=optimizer_dict.keys())
    parser.add_argument('--optim-wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim-mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim-nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n-epochs', type=int,
                        help='number of epochs.')
    parser.add_argument('--epoch-scaling', type=str, default='const',
                        help='How to scale the epochs for the current domain training.', choices=['const', 'linear', 'sqrt'])
    parser.add_argument('--batch-size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=3407,
                        help='The random seed.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='The number of workers.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--visualize', action='store_true', help='whether return all the data and prediction.')
    parser.add_argument('--non-verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--validation-perc', type=float, default=0.1,
                        help='percentage of the validation data.')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Whether checkpoint the model backbone parameters.')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug-mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', action='store_true', help='Inhibit wandb logging')
    parser.add_argument('--wandb-entity', type=str, default='haizhou-shi', help='Wandb entity') # change it to your own wandb account.
    parser.add_argument('--wandb-project', type=str, default='UDIL', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='', help="Wandb run's name")
    


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer-size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--buffer-batch-size', type=int,
                        help='The batch size of the memory buffer.')


def add_backbone_args(parser: ArgumentParser) -> None:
    parser.add_argument('--backbone', type=str, required=True,
                        help='Backbone name.', choices=get_all_backbones())
    parser.add_argument('--hiddim', type=int, default=800,
                        help='name of the encoder.') 
