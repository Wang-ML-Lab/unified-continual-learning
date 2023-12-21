import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier

import ipdb

class Resnet18(FwdContinualBackbone):
    NAME = 'resnet18'
    def __init__(self, indim, hiddim, outdim, args):
        super(Resnet18, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

        # dictionary for the output and the intermediate layers
        self.fwd_outputs = {}

        def get_feature(name):
            def hook(model, input, output):
                self.fwd_outputs[name] = output
            return hook

        self.net = resnet18(num_classes=outdim)

        # forward hook to store the intermediate results
        self.net.avgpool.register_forward_hook(get_feature('features'))
        self.net.fc.register_forward_hook(get_feature('logits'))

    def forward(self, x, returnt='logits'):
        self.net(x)
        if returnt == 'features':
            shape = self.fwd_outputs['features'].shape
            return self.fwd_outputs['features'].view(*shape[:2])
        elif returnt == 'logits':
            return self.fwd_outputs['logits']
        elif returnt == 'prob':
            return self.softmax(self.fwd_outputs['logits'])
        elif returnt == 'all':
            logits = self.fwd_outputs['logits']
            probs = self.softmax(self.fwd_outputs['logits'])

            shape = self.fwd_outputs['features'].shape
            features = self.fwd_outputs['features'].view(*shape[:2])
            return logits, probs, features


if __name__ == '__main__':
    model = Resnet18(num_classes=50)
    x = torch.ones([2, 3, 128, 128])
    
    ipdb.set_trace()
    