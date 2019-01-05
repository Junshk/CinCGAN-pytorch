import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        self.benchmark = False
        apath = args.testpath + '/' + args.testset + '/x' + str(args.scale[0])

        self.filelist = []
        self.imnamelist = []
        if not train:
            for f in os.listdir(apath):
                try:
                    filename = os.path.join(apath, f)
                    misc.imread(filename)
                    self.filelist.append(filename)
                    self.imnamelist.append(f)
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename_hr = os.path.join('../HR/{}/x{}'.format(self.args.testset, str(self.args.scale[0])), self.imnamelist[idx].replace('LRBI', 'HR'))
        filename, _ = os.path.splitext(filename)


        hr = misc.imread(filename_hr)
        hr = common.set_channel([hr], self.args.n_colors)[0]

        lr = misc.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return common.np2Tensor([lr], self.args.rgb_range)[0], common.np2Tensor([hr], self.args.rgb_range)[0], filename
    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

