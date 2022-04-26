import logging
import torch

from .resnet_3d import RESNET18  # This require Pytorch >= 1.2.0 support
from .rotnet_resnet import resnet18
from .rotnet_classifier import RotnetClassifier
from .config import get_config

def get_symbol(name, print_net=False, **kwargs):
	
	if name.upper() == "R3D18":
		net = RESNET18(**kwargs)
	elif name == "rotnet":
		net = resnet18(
			num_classes = 4,
			shortcut_type = 'A', 
			sample_size=kwargs["sample_size"], 
			sample_duration=kwargs["sample_duration"]
		).cuda()
	elif name == "rotnet_classifier":
		net = RotnetClassifier(
			**kwargs
		)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

