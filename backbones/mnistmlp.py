import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier

# The creation of the backbones follows the following paradigm:
#   Backbone(indim, hiddim, outdim, args)

class LinearClassifier(nn.Module):
    """Linear classifier, predicting the class label."""
    NAME = 'mnist-classifier'
    def __init__(self, indim, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(indim, num_classes)

    def forward(self, x, return_softmax=False):
        x = self.classifier(x)
        x_softmax = F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)

        return x, x_softmax if return_softmax else x


class MNISTMLP(FwdContinualBackbone):
    NAME = 'mnistmlp'
    
    def __init__(self, indim, hiddim, outdim, args) -> None:
        super().__init__()
        self.indim = indim # indim not necessarily an integer
        self.hiddim = hiddim
        self.outdim = outdim
        
        # constructing the encoder and the predictor
        if isinstance(indim, tuple) or isinstance(indim, list):
            self.enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(indim), hiddim),
                nn.ReLU(),
                nn.Linear(hiddim, hiddim),
                nn.ReLU()
            )
        else: 
            self.enc = nn.Sequential(
                nn.Linear(indim, hiddim),
                nn.ReLU(),
                nn.Linear(hiddim, hiddim),
                nn.ReLU()
            )

        self.pred = LinearClassifier(indim=hiddim, num_classes=outdim)

        self.net = nn.Sequential(self.enc, self.pred)
        self.reset_parameters()

    def forward(self, x, returnt='logits'):
        feats = self.enc(x)
        
        if returnt == 'features':
            return feats
        
        # classifier supports returning two outputss
        logits, prob = self.pred(feats, return_softmax=True)

        if returnt == 'logits':
            return logits
        elif returnt == 'prob':
            return prob
        elif returnt == 'all':
            return logits, prob, feats
        else:
            return NotImplementedError("Unsupported return type")


    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)