"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import ResNet as resnet


class BaseEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.feature_size = 2048
        self.outsize = outsize
        # self.encoder = resnet.load_ResNet50Model()  # out: 2048
        self._create_encoder()
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outsize)
        )
        self.last_op = last_op

    def forward(self, inputs, output_features=False):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        if not output_features:
            return parameters
        return parameters, features

    def _create_encoder(self):
        raise NotImplementedError()


class ResnetEncoder(BaseEncoder):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__(outsize, last_op)

    def _create_encoder(self):
        self.encoder = resnet.load_ResNet50Model()  # out: 2048


class SecondHeadResnet(nn.Module):

    def __init__(self, enc : BaseEncoder, outsize, last_op=None):
        super().__init__()
        self.resnet = enc # yes, self.resnet is no longer accurate but the name is kept for legacy reasons (to be able to load old models)
        self.layers = nn.Sequential(
            nn.Linear(self.resnet.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        if last_op == 'same':
            self.last_op = self.resnet.last_op
        else:
            self.last_op = last_op


    def forward(self, inputs):
        out1, features = self.resnet(inputs, output_features=True)
        out2 = self.layers(features)
        if self.last_op:
            out2 = self.last_op(out2)
        return out1, out2


    def train(self, mode: bool = True):
        #here we NEVER modify the eval/train status of the resnet backbone, only the FC layers of the second head
        self.layers.train(mode)
        return self