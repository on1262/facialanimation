from transformers import Wav2Vec2Model, Wav2Vec2Config
import librosa
import numpy as np
import torch
import torch.optim as optim
import os
from os.path import join
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from ERDataset import EnsembleDataset, EREnsembleCollate_fn, CREMADDataset
from Extractor import ExtractorRevGrad
import platform
import math

if __name__ == "__main__":
  batch_size = 6
  num_epoch = 100
  emotion_class= 6
  imbalance_sample = True
  alpha = 0.3
  pre_load = True
  debug_flag = False
  device=torch.device('cuda:3')
  # load pretrained model
  root_path_windows = r"F:\Project\FacialAnimation\facialanimation\wav2vec2"
  root_path_linux = r"/home/chenyutong/facialanimation/wav2vec2"
  root_path = root_path_linux
  if platform.system() == "Windows":
    root_path = root_path_windows
  tf_extractor = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, config=join(root_path, 'pretrained_model','config.json'), state_dict=torch.load(join(root_path,'pretrained_model','pytorch_model.bin')))

  additional_extractor = ExtractorRevGrad(emotion_class=emotion_class, dataset_label=2, alpha=alpha)

  # loading dataset
  IEMOCAProot = r'/mnt/lv2/data/ser/IEMOCAP'
  IEMOCAP_index_path = join(root_path,'iemocap.csv')
  label_list = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1} # max label < emotion_class
  path1_preload = join(root_path,'RAVDESS16kHz.pt')
  path2_preload = join(root_path,'CREMAD16kHz.pt')
  path1 = join(root_path, 'dataset_cache', 'RAVDESS')
  path2 = join(root_path, 'dataset_cache', 'CREMA-D')
  data_path = None
  if pre_load is True:
    data_path = {'RAV': path1_preload, 'CRE':path2_preload, 'IEM':IEMOCAProot}
  else:
    data_path = {'RAV': path1, 'CRE':path2, 'IEM':IEMOCAProot}
  dataset_train = EnsembleDataset(data_path, IEMOCAP_index_path, label_list,dataset_type='all', max_len_s=2.5,pre_load=pre_load, debug_flag=debug_flag)
  dataset_test = CREMADDataset(data_path['CRE'], label_list, dataset_type='all', pre_load=pre_load, debug_flag=debug_flag)
  train_num = len(dataset_train)
  test_num = len(dataset_test)
  
  train_loader = None
  # imbalance sampler for train dataset
  if imbalance_sample:
    labels = [0 for _ in range(emotion_class)]
    for idx in range(len(dataset_train)):
      _, data = dataset_train[idx]
      _, label = data
      labels[label.item()] += 1
    weights = []
    labels = [1/i for i in labels]
    for idx in range(len(dataset_train)):
      _, data = dataset_train[idx]
      _, label = data
      weights.append(labels[label.item()])
    sampler = WeightedRandomSampler(weights, len(dataset_train))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, collate_fn=EREnsembleCollate_fn)
  else:
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, collate_fn=EREnsembleCollate_fn)


  criterion = torch.nn.CrossEntropyLoss(reduction='mean')
  criterion_dom = torch.nn.CrossEntropyLoss(reduction='mean', weight=dataset_train.get_class_weight().to(device))
  opt = optim.Adam(additional_extractor.parameters())
  opt_tf = optim.Adam(tf_extractor.parameters())

  additional_extractor = additional_extractor.to(device)
  tf_extractor = tf_extractor.to(device)




  print('alpha=', alpha)
  best_test_acc = 0
  for epoch in range(num_epoch):
    # adjust learning rate
    for g in opt.param_groups:
      g['lr'] = (1e-3)/(epoch+1) + 1e-4
    for g in opt_tf.param_groups:
      g['lr'] = (2e-5)/(epoch+1) + 1e-5

    print("epoch:", epoch, " train")
    tf_extractor.train()
    additional_extractor.train()
    
    acc = 0
    acc_table = [0 for _ in range(emotion_class)]
    class_table = [0 for _ in range(emotion_class)]
    for (iter,data) in enumerate(train_loader):
      dom, data_tensor,label = data['domain'], data['x'], data['y']
      dom, data_tensor,label = dom.to(device), data_tensor.to(device), label.to(device)
      out = tf_extractor(data_tensor, output_hidden_states=True)
      tuples = out.hidden_states
      h_layers = torch.stack(tuples, dim=1) #(batch_size, 13, sequence_length, hidden_size).
      # out
      # last_hidden_state: ([1, len, 768]) 
      # extract_features: ([1, len, 512])
      #print(out.last_hidden_state.size(), out.extract_features.size())
      out2 = additional_extractor(h_layers)
      out_label, out_dom = out2['label'], out2['domain']
      # loss & backward
      loss_label = criterion(out_label, label) 
      loss_dom = criterion_dom(out_dom, dom)
      loss = loss_label + loss_dom
      opt.zero_grad()
      opt_tf.zero_grad()
      loss.backward()
      opt.step()
      opt_tf.step()
      # calculate acc
      d_label = label.detach().cpu()
      d_out = out_label.detach().cpu()
      for k in range(d_label.size(0)):
        class_table[d_label[k]] += 1
        if torch.argmax(d_out[k,:]) == d_label[k]:
          acc += (1/train_num)
          acc_table[d_label[k]] += 1
      
      if iter % round(len(train_loader)/10) == 0 and iter != 0:
        print("train iter=", iter, " accuracy=", acc*train_num/(((iter+1)*batch_size)), " loss_label=", loss_label.item(), ' log(-loss_dom)=', math.exp(-loss_dom.item()))
    acc_table = [acc_table[i]/class_table[i] if class_table[i] > 0 else 0 for i in range(emotion_class)]
    print("epoch=", epoch, "  train acc=", acc, " avg recall=", sum(acc_table)/emotion_class)
    print('mat avg distribution:', additional_extractor.get_m().detach().cpu())
    
    
    print('testing')

    tf_extractor.eval()
    additional_extractor.eval()
    acc = 0
    acc_table = [0 for _ in range(emotion_class)]
    class_table = [0 for _ in range(emotion_class)]

    for iter in range(test_num):
      data_tensor,label = dataset_test[iter] # may not cover all samples TODO change when dataset is EnsembleDataset
      data_tensor,label = data_tensor.to(device), label.to(device)
      out = tf_extractor(data_tensor, output_hidden_states=True)
      tuples = out.hidden_states 
      h_layers = torch.stack(tuples, dim=1) #(batch_size, 13, sequence_length, hidden_size).
      out2 = additional_extractor(h_layers)
      out2 = out2['label']

      # calculate acc
      d_label = label.detach().cpu()
      class_table[d_label[0]] += 1
      if torch.argmax(out2.detach().cpu(),dim=1) == d_label:
        acc += (1/test_num)
        acc_table[d_label[0]] += 1
      iter += 1
      if iter % round(test_num/5) == 0:
        print("test iter=", iter, " accuracy=", acc*test_num/(iter+1))
    acc_table = [acc_table[i]/class_table[i] if class_table[i] > 0 else 0 for i in range(emotion_class)]
    print("epoch=", epoch, "  test acc=", acc, " avg recall=", sum(acc_table)/emotion_class)
    if acc > best_test_acc:
      best_test_acc = acc
      print('best model saved')
      torch.save({'extractor': additional_extractor.state_dict(), 'transformer':tf_extractor.state_dict()} , 'best_model.pt')