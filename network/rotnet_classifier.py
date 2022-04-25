import os
import torch
from .rotnet_resnet import resnet18

try:
	from . import initializer
	from .utils import load_state
except: 
	import initializer
	from utils import load_state

class RotnetClassifier(torch.nn.Module):

    def __init__(self, num_classes, pretrained_path=None, **kwargs):
        super(RotnetClassifier, self).__init__()

        # setup pretrained SSL Rotnet backbone
        self.rotnet_resnet = resnet18(
            num_classes=4,
            shortcut_type='A',
            sample_size=kwargs["sample_size"],
            sample_duration=kwargs["sample_duration"]
        ).cuda()

        assert os.path.exists(pretrained_path), f"cannot locate:{pretrained_path}"
        pretrained_model = torch.load(pretrained_path)
        load_state(self.rotnet_resnet, pretrained_model)

        # freeze backbone
        for param in self.rotnet_resnet.parameters():
            param.requires_grad = False

        self.rotnet_resnet.fc2 = torch.nn.Identity()

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )


    def forward(self, x):

        x = self.rotnet_resnet(x)
        y = self.classifer(x)

        return y
