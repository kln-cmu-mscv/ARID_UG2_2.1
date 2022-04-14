import os
import logging
import torch
import torch.nn as nn
import numpy as np
import random


from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter
from torch.utils.data.sampler import Sampler

def get_arid(data_root='./dataset/ARID', clip_length=8, train_interval=2,
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
			seed=0, **kwargs):
	""" data iter for ARID
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}], seed = {}".format(clip_length, train_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'train_data'),
					  csv_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID1.1_t1_train_pub.csv'),
					  sampler=train_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]),
										 transforms.RandomCrop((224, 224)), # insert a resize if needed
										 transforms.RandomHorizontalFlip(),
										 transforms.RandomHLS(vars=[15, 35, 25]),
										 transforms.ToTensor(),
										 normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	return train

class ClasswiseBatchSampler(Sampler):
	def __init__(self, dataset, batch_size):	
		len_dataset = len(dataset)  
		self.dataset = dataset
		self.batch_size = batch_size

	def __iter__(self):
		samples = self.get_samples(self.dataset, self.batch_size)
		#random.shuffle(samples)
		return iter(samples)

	def __len__(self):
		return len(self.get_samples(self.dataset, self.batch_size))
		
	def get_samples(self,train, batch_size):

		class_inds = [torch.where(train.labels == class_idx)[0]
				for class_idx in torch.unique(train.labels)]

		samples = []
		for i in range(len(class_inds)):
			sample = torch.randint(0, len(class_inds[i]), size = (len(class_inds[i])//batch_size, batch_size))
			samples.extend(class_inds[i][sample].tolist())
		
		return samples
		

def creat(name, batch_size, num_workers=8, **kwargs):

	if name.upper() == 'ARID':
		train = get_arid(**kwargs)
	else:
		assert NotImplementedError("iter {} not found".format(name))

	custom_sampler = ClasswiseBatchSampler(train, batch_size)
	train_loader = torch.utils.data.DataLoader(train, batch_sampler = custom_sampler,shuffle=False, num_workers=num_workers, pin_memory=False)
	
	return train_loader
