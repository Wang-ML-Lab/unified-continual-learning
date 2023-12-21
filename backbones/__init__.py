import os
import importlib

import math 
import torch
import torch.nn as nn

def get_all_backbones():
    return [backbone.split('.')[0] for backbone in os.listdir('backbones')
            if not backbone.find('__') > -1 and 'py' in backbone]

names = {}
for backbone in get_all_backbones():
    mod = importlib.import_module('backbones.' + backbone)
    class_name = {x.lower():x for x in mod.__dir__()}[backbone.replace('_', '')]
    names[backbone] = getattr(mod, class_name)

def get_backbone(backbone_name, indim, hiddim, outdim, args):
    """Get the network architectures for encoder, predictor, discriminator."""
    return names[backbone_name](indim, hiddim, outdim, args)

