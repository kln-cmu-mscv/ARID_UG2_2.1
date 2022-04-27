
# Require Pytorch Version >= 1.2.0
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

try:
    from . import initializer
    from .utils import load_state
except: 
    import initializer
    from utils import load_state


class ARIDRotnetClassifier(torch.nn.Module):

    def __init__(self, num_classes, pretrained_path=None, **kwargs):
        super(ARIDRotnetClassifier, self).__init__()

        # setup pretrained SSL Rotnet backbone
        self.resnet3d = torchvision.models.video.r3d_18(pretrained=False, progress=False, num_classes=4, **kwargs)

        ###################
        # Initialization #
        initializer.xavier(net=self)

        assert os.path.exists(pretrained_path), f"cannot locate:{pretrained_path}"
        pretrained_model_metadata = torch.load(pretrained_path)
        state_load_result = load_state(self.resnet3d, pretrained_model_metadata['state_dict'], load_fc=False)

        if state_load_result: print("successfully loaded pretrained model.")

        # freeze backbone until last layer
        for param in list(self.resnet3d.parameters())[:-1]:
            param.requires_grad = False

        self.resnet3d.fc = torch.nn.Linear(
            self.resnet3d.fc.in_features,
            num_classes
        )

    def forward(self, x):

        h = self.backbone(x)

        return h
