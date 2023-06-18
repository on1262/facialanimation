import json
import os, sys
import torch
sys.path.append('/home/chenyutong/facialanimation')

def check_params_distribution(dataset):
    distri = torch.zeros(56)
    for data in dataset:
        params = data['params']
        distri += torch.mean(torch.abs(params), dim=0)
    return distri / len(dataset)
