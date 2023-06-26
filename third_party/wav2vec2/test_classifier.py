from transformers import Wav2Vec2Model, Wav2Vec2Config
import librosa
import numpy as np
import torch
import torch.optim as optim
import os
from os.path import join
import random
from ERDataset import RAVDESSDataset, CREMADDataset,IEMOCAPDataset, ERCollate_fn
from Extractor import ExtractorLSTM
import platform

if __name__ == "__main__":
  emotion_class= 6
  # load pretrained model
  root_path_windows = r"F:\Project\FacialAnimation\facialanimation\wav2vec2"
  root_path_linux = r"/home/chenyutong/project/FacialAnimation/wav2vec2"
  root_path = root_path_linux
  if platform.system() == "Windows":
    root_path = root_path_windows

  model = torch.load(join(root_path,'best_model.pt'))  
  tf_extractor = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, config=join(root_path, 'pretrained_model','config.json'), state_dict=torch.load(join(root_path,'pretrained_model','pytorch_model.bin')))
  tf_extractor.load_state_dict(model['transformer'].state_dict())
  additional_extractor = ExtractorLSTM(emotion_class=6)
  additional_extractor.load_state_dict(model['extractor'].state_dict())
  
  print('model loaded')

  # loading dataset
  IEMOCAProot = r'/mnt/lv2/data/ser/IEMOCAP'
  IEMOCAP_index_path = join(root_path,'iemocap.csv')
  label_list = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1} # max label < emotion_class

  print('label_list:', label_list)
  dataset_test = [IEMOCAPDataset(IEMOCAProot,IEMOCAP_index_path, label_list,dataset_type='test', in_memory=True),\
     RAVDESSDataset(join(root_path, 'RAVDESS16kHz.pt'), label_list, pre_load=True),\
         CREMADDataset(join(root_path,  'CREMAD16kHz.pt'), label_list, pre_load=True)]
  test_num = [len(dataset) for dataset in dataset_test]
  dataset_name = ['IEMOCAP', 'RAVDESS', 'CREMAD']

  additional_extractor = additional_extractor.cuda()
  tf_extractor = tf_extractor.cuda()


  for (idx, dataset) in enumerate(dataset_test):
    print('testing dataset ', dataset_name[idx])

    tf_extractor.eval()
    additional_extractor.eval()
    acc = 0
    acc_table = [0 for _ in range(emotion_class)]
    class_table = [0 for _ in range(emotion_class)]

    for iter in range(test_num[idx]):
      data_tensor,label = dataset[iter] # may not cover all samples
      data_tensor,label = data_tensor.cuda(), label.cuda()
      out = tf_extractor(data_tensor, output_hidden_states=True)
      tuples = out.hidden_states 
      h_layers = torch.stack(tuples, dim=1) #(batch_size, 13, sequence_length, hidden_size).
      out2 = additional_extractor(h_layers)

      # calculate acc
      d_label = label.detach().cpu()
      class_table[d_label[0]] += 1
      if torch.argmax(out2.detach().cpu(),dim=1) == d_label:
        acc += (1/test_num[idx])
        acc_table[d_label[0]] += 1
      iter += 1
      if iter % round(test_num[idx]/5) == 0:
        print("test iter=", iter, " accuracy=", acc*test_num[idx]/(iter+1))
    acc_table = [acc_table[i]/class_table[i] if class_table[i] > 0 else 0 for i in range(emotion_class)]
    print("dataset=", dataset_name[idx], "  test acc=", acc, " avg recall=", sum(acc_table)/emotion_class)