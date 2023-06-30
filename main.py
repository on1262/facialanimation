from trainer import Trainer
from inference import Inference
from utils.config_loader import GBL_CONF, PATH
import argparse
import os
import torch
import numpy as np
import random

def init_environ():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='train or inference')
    parser.add_argument('--scripts', default='', type=str, help='run scripts')
    args = parser.parse_args()

    init_environ()
    seed_torch(GBL_CONF['global']['seed'])
    # max_batch_size_dict = {'emo': 8, 'lstm_style' : 128, 'tf' : 70}
    
    if args.mode == 'train':
        trainer = Trainer()
        trainer.run_epochs()
    elif args.mode == 'inference':
        inference = Inference()
        inference.inference()
    elif args.mode == 'scripts':
        if args.scripts == 'dataset_cache':
            from scripts.dataset_cache import dataset_cache
            dataset_cache()

    

