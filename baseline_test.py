import sys
import os
import torch
from torch.utils.data import Dataset
from utils.interface import LSTMEMO, FaceFormerModel, VOCAModel, BaselineConverter
from dataset import BaselineBIWIDataset, BaselineVOCADataset
from utils.fitting.fit import Mesh, approx_transform, approx_transform_mouth, get_mouth_landmark
import torch.nn.functional as F
import subprocess
import argparse
import numpy as np
from utils.detail_fixer import DetailFixer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tf_emo_4',)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_obj', type=bool, default=False)
    args = parser.parse_args()
    
    # load biwi test dataset
    device = torch.device('cuda:' +str(args.device))
    test_voca_path = r'/home/chenyutong/facialanimation/dataset_cache/VOCASET'
    label_dict = None
    
    # load model
    if 'emo' in args.model:
        model = LSTMEMO(device, model_name=args.model)
        model_output_type = 'flame'
    elif args.model == 'convert':
        model = BaselineConverter(device)
        model_output_type = 'flame'
    elif args.model == 'faceformer_flame':
        model = FaceFormerModel(device)
        model_output_type = 'flame'
    elif args.model == 'voca':
        model = VOCAModel(device)
        model_output_type = 'flame'
    baseline_test(args.model, model, model_output_type, dataset=test_dataset, gt_output_type='flame', device=device, save_obj=args.save_obj)