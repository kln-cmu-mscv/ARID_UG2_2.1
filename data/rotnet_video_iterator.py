import numpy as np
import torch
from .video_iterator import VideoIter

class RotnetVideoIter(VideoIter):
    def __init__(self, **kwargs):
        super(RotnetVideoIter, self).__init__(**kwargs)


    def __getitem__(self, index):
        if self.return_item_subpath:
            clip_input, label, vid_subpath = super().__getitem__(index)
        else:
            clip_input, label = super().__getitem__(index)

        clip_one = clip_input.numpy()
        clip_two = np.rot90(clip_one, k = 1 , axes = (2,3))
        clip_three = np.rot90(clip_one, k = 2 , axes = (2,3))
        clip_four = np.rot90(clip_one, k = 3 , axes = (2,3))

        clip_one = torch.from_numpy(clip_one.copy())
        clip_two = torch.from_numpy(clip_two.copy())
        clip_three = torch.from_numpy(clip_three.copy())
        clip_four = torch.from_numpy(clip_four.copy())

        target_one = torch.LongTensor(np.array([0]))
        target_two = torch.LongTensor(np.array([1]))
        target_three = torch.LongTensor(np.array([2]))
        target_four = torch.LongTensor(np.array([3]))

        clip = torch.cat((clip_one, clip_two, clip_three, clip_four), dim=0)
        target = torch.cat((target_one, target_two, target_three, target_four), dim=0)

        if self.return_item_subpath:
            return clip, target, vid_subpath
        else:
            return clip, target