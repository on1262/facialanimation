from trainer import Trainer
import argparse
import traceback
import sys
import time
import glob
import os
import torch
import gc
import numpy as np
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def seed_torch(seed=1000):
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
    parser.add_argument('--device', type=int, default=1, nargs='+', help='cuda device, integer')
    parser.add_argument('--debug', type=int, nargs='?', default=0, choices=[0,1,2], help='debug mode, 0=no debug, 1=check code, 2=mono batch check')
    parser.add_argument('--model', type=str, default='cnn_lstm', help='run model type')
    parser.add_argument('--epoch', type=int, default=30, help='epoch')
    parser.add_argument('--version', type=int, default=0, help='change model hyper-parameter configs, default 0')
    parser.add_argument('--log', type=bool, default=False, help='log sample name for visualization')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    
    args = parser.parse_args()
    seed_torch()
    max_batch_size_dict = {'emo': 8, 'lstm_style' : 128, 'tf' : 70}
    for key, val in max_batch_size_dict.items():
        if key in args.model and args.batch_size > val:
            print('Warning: batch_size larger than ', val, ' for model type ', key, ' may cause cuda run out of memory')

    cre_path = '/home/chenyutong/facialanimation/dataset_cache/CREMA-D'
    lrs2_path = '/home/chenyutong/facialanimation/dataset_cache/LRS2'
    biwi_path = '/home/chenyutong/facialanimation/dataset_cache/BIWI'
    vocaset_path = '/home/chenyutong/facialanimation/dataset_cache/VOCASET'
    
    dataset_path = {'CRE': cre_path, 'LRS2':lrs2_path, 'BIWI':biwi_path, 'VOCASET':vocaset_path}
    wav2vec_path = r'F:\Project\FacialAnimation\facialanimation\wav2vec2\pretrained_model' if sys.platform == 'win32' \
        else '/home/chenyutong/facialanimation/wav2vec2/pretrained_model'
    model_name = args.model
    model_path = '/home/chenyutong/facialanimation/Model/' + model_name

    # handle device
    if isinstance(args.device, list):
        dev_str = ['cuda:' + str(d) for d in args.device]
    else:
        dev_str = ['cuda:' + str(args.device)]
    
    # config memory use
    trainer = Trainer(imbalance_sample=False if args.debug > 0 else True, 
        load_path=None,
        save_path=model_path,
        wav2vec_path=wav2vec_path,
        dataset_path=dataset_path,
        model_name=model_name,
        label_dict={'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1},
        dev_str=dev_str,
        version=args.version,
        batch_size=args.batch_size,
        log_samples=True if args.debug == 0 else args.log,
        debug=args.debug
        )

    if args.debug == 0:
        best_loss = None
        last_save_path = None
        reload_path = None
        for epoch in range(1,args.epoch+1):
            gc.collect()
            pre_ticks = time.time()
            try:
                avg_loss = trainer.run_epoch()
                post_ticks = time.time()
                mins = (post_ticks - pre_ticks) / 60
                print('epoch >> ', round(mins), ' mins ', end='')
                best_loss = avg_loss if best_loss is None or avg_loss < best_loss else best_loss
                print('best loss=', best_loss)
                last_save_path = trainer.save()
            except RuntimeError as e:
                print(traceback.format_exc())
                print('='*20, 'cuda memory summary', '='*20)
                print(torch.cuda.memory_summary(torch.device(dev_str[0])))
                print('='*20, 'END', '='*20)
                del trainer.model
                gc.collect()
                with torch.cuda.device(torch.device(dev_str[0])):
                    torch.cuda.empty_cache()

                if last_save_path is not None:
                    if reload_path == last_save_path: # never reload again
                        print('avoid reload again, exit.')
                        break
                    else:
                        reload_path = last_save_path
                else:
                    list_of_files = glob.glob(os.path.join(model_path, 'saved_model', '*.pth'))
                    latest_file_path = max(list_of_files, key=os.path.getctime)
                    if reload_path == latest_file_path: # never reload again
                        print('avoid reload again, exit.')
                        break
                    else:
                        reload_path = latest_file_path
                print('reload model from: ', reload_path)
                trainer.load(reload_path)
    else:
        best_loss = None
        for epoch in range(1, args.epoch+1):
            avg_loss = trainer.run_epoch()
            best_loss = avg_loss if best_loss is None or avg_loss < best_loss else best_loss
            print('best loss=', best_loss)
            if epoch != 0 and epoch % 70 == 0:
                trainer.save()
            if args.debug == 2:
                print('debug mode 2, single epoch stopped')
                break

