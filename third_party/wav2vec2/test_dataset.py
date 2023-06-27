import os
from ERDataset import RAVDESSDataset, CREMADDataset,IEMOCAPDataset
import librosa
from os.path import join
import random
import numpy as np
import torch
import soundfile as sf
import platform

print('testing dataset')
root_path_windows = r"third_party/wav2vec2"
root_path_linux = r"third_party/wav2vec2"
root_path = root_path_linux
if platform.system() == "Windows":
  root_path = root_path_windows


# loading dataset
IEMOCAProot = r'/mnt/lv2/data/ser/IEMOCAP'
IEMOCAP_index_path = join(root_path,'iemocap.csv')
label_list = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1} # max label < emotion_class
#path1 = join(root_path,'RAVDESS16kHz.pt')
#path2 = join(root_path,'CREMAD16kHz.pt')
#dataset_train = IEMOCAPDataset(IEMOCAProot,IEMOCAP_index_path, label_list,dataset_type='train', in_memory=True)
dataset_test = [IEMOCAPDataset(IEMOCAProot,IEMOCAP_index_path, label_list,dataset_type='test', in_memory=True),\
    RAVDESSDataset(join(root_path, 'RAVDESS16kHz.pt'), label_list, pre_load=True),\
        CREMADDataset(join(root_path, 'CREMAD16kHz.pt'), label_list, pre_load=True)]
test_num = [len(dataset) for dataset in dataset_test]
dataset_name = ['IEMOCAP', 'RAVDESS', 'CREMAD']

for (idx,dataset) in enumerate(dataset_test):
    for k in label_list.keys():
        data = None
        count = 0
        while data is None and count < 2000:
            count+=1
            (d, label) = random.choice(dataset)
            if label.item() == label_list[k]:
                data = d
        if data is not None:
            print(join(root_path,'wav_cache',dataset_name[idx]+'_'+k+'.wav'))
            sf.write(join(root_path,'wav_cache',dataset_name[idx]+k+'.wav'), data.squeeze(0).numpy(), 16000)

print('Done')

