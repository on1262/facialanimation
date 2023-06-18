import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import librosa
import os
import csv
from os.path import join
import random

def ERCollate_fn(data:list):
    data_list = []
    label_list = []
    max_size = 0
    for (d,label) in data:
        data_list.append(d)
        label_list.append(label)
        if d.size(1) > max_size:
            max_size = d.size(1)
    for (idx,d) in enumerate(data_list):
        data_list[idx] = F.pad(d, (0,max_size-d.size(1)),mode='constant',value=0)
    return {'x': torch.stack(data_list,dim=0).squeeze(dim=1),  'y': torch.stack(label_list,dim=0).squeeze(dim=1)} #(batch_size, max_sequence_len)

def EREnsembleCollate_fn(data:list):
    data_list = []
    label_list = []
    ind_list = []
    max_size = 0
    for (ind,(d,label)) in data:
        data_list.append(d)
        label_list.append(label)
        ind_list.append(ind)
        if d.size(1) > max_size:
            max_size = d.size(1)
    for (idx,d) in enumerate(data_list):
        data_list[idx] = F.pad(d, (0,max_size-d.size(1)),mode='constant',value=0)
    return {'x': torch.stack(data_list,dim=0).squeeze(dim=1), \
         'y': torch.stack(label_list,dim=0).squeeze(dim=1),\
            'domain': torch.stack(ind_list,dim=0).squeeze(dim=1)} #(batch_size, max_sequence_len)


class RAVDESSDataset(Dataset):
    def __init__(self, path, label_list: dict, dataset_type='train', pre_load=False, debug_flag=False):
        super(RAVDESSDataset, self).__init__()
        self.dict = {'01':'NEU','02':'CAL','03':'HAP','04':'SAD','05':'ANG','06':'FEA','07':'DIS','08':'SUR'}
        self.label_list = label_list
        print('Loading RAVDESS Dataset')
        self.datalist = []
        hours = 0
        if pre_load is True and debug_flag is False:
            self.datalist = torch.load(path)
        else:
            debug_count = 0
            for root, _, files in os.walk(path, topdown=False):
                for name in files:
                    # get label from name
                    emo = self.dict[name.split('-')[2]]
                    if self.label_list.get(emo) is not None:
                        data, sr = librosa.load(os.path.join(root,name),sr=16000)
                        data_tensor = torch.from_numpy(data.T)
                        data_tensor = torch.unsqueeze(data_tensor, 0)
                        self.datalist.append((data_tensor, emo))
                        hours += data_tensor.size(1)/16000
                        debug_count += 1
                        if debug_count >= 100 and debug_flag is True:
                            print('debug mode, return as count=', debug_count)
                            return
            print('RAVDESS load ', hours/3600, 'hours')
            torch.save(self.datalist, 'RAVDESS16kHz.pt')
        offset = round(0.8*len(self.datalist))
        if dataset_type == 'train':
            self.datalist = self.datalist[0:offset]
        elif dataset_type == 'test':
            self.datalist = self.datalist[offset:]
        else:
            offset = len(self.datalist)
        print('setting offset=', offset,' for ', dataset_type, ' mode')


    
    def __getitem__(self, index):
        dt, emo = self.datalist[index]
        return (dt, torch.LongTensor(data=[self.label_list[emo]]))

    def __len__(self):
        return len(self.datalist)

class CREMADDataset(Dataset):
    def __init__(self, path, label_list: dict,dataset_type='train', pre_load=False, debug_flag=False):
        super(CREMADDataset, self).__init__()
        self.label_list = label_list
        self.dict = {'ANG':'ANG','DIS':'DIS','FEA':'FEA','HAP':'HAP','NEU':'NEU','SAD':'SAD'}
        print('Loading CREMAD Dataset')
        self.datalist = []
        if pre_load is True and debug_flag is False:
            self.datalist = torch.load(path)
        else:
            hours = 0
            debug_count = 0
            for root, _, files in os.walk(path, topdown=False):
                for name in files:
                    # get label from name
                    emo = name.split('_')[2]
                    if self.dict[emo] is not None and self.label_list.get(self.dict[emo]) is not None:
                        data, sr = librosa.load(os.path.join(root,name),sr=16000)
                        data_tensor = torch.from_numpy(data.T)
                        data_tensor = torch.unsqueeze(data_tensor, 0)
                        self.datalist.append((data_tensor, emo))
                        hours += data_tensor.size(1)/16000
                        debug_count += 1
                        if debug_count > 100 and debug_flag is True:
                            print('debug mode, return as count=', debug_count)
                            return
            print('CREMAD load ', hours/3600, 'hours')
            torch.save(self.datalist, 'CREMAD16kHz.pt')
        offset = round(0.8*len(self.datalist))
        if dataset_type == 'train':
            self.datalist = self.datalist[0:offset]
        elif dataset_type == 'test':
            self.datalist = self.datalist[offset:]
        print('setting offset=', offset,' for ', dataset_type, ' mode')
    
    def __getitem__(self, index):
        dt, emo = self.datalist[index]
        return (dt, torch.LongTensor(data=[self.label_list[emo]]))

    def __len__(self):
        return len(self.datalist)

class IEMOCAPDataset(Dataset):
    def __init__(self, data_path, index_path, label_list: dict, dataset_type='train', in_memory=True,max_len_s=4, debug_flag=False):
        super(IEMOCAPDataset, self).__init__()
        self.dict = {'neu':'NEU','hap':'HAP','sad':'SAD','ang':'ANG','fru':'FRU', 'sur':'SUR', 'exc':'EXC', 'dis':'DIS', 'fea':'FEA'} #9 emotions
        self.label_list = label_list
        self.max_len = round(max_len_s*16000)
        print('Loading IEMOCAP Dataset',' in_memory=', in_memory)
        self.in_memory = in_memory
        self.data_name_list = []
        self.datalist = []
        
        # read meta data (all session)
        with open(index_path, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for row in data:
                p = join(data_path, row[6])
                if self.dict.get(row[3]) is not None and self.label_list.get(self.dict[row[3]]) is not None:
                    label = self.label_list[self.dict[row[3]]] # label_list should not be fixed in pre load dataset
                    self.data_name_list.append((p, label))
        offset = round(0.8*len(self.data_name_list))
        if dataset_type != 'all':
            self.load_list = self.data_name_list[0:offset] if dataset_type == 'train' else self.data_name_list[offset:]
        else:
            self.load_list = self.data_name_list
        file_count = 0
        if dataset_type == 'train':
            file_count = offset  
        elif dataset_type == 'test':
            file_count = len(self.data_name_list) - offset
        else:
            file_count = len(self.data_name_list)
        print('IEMOCAP load ', file_count,' files')
        if not in_memory:
            return
        
        # read wavs
        hours = 0
        print('IEMOCAP: loading wavs')
        
        if debug_flag is True:
            print('debug mode, modify load list size=100')
            self.load_list = self.load_list[0:100]

        for (p, label) in self.load_list:
            label = torch.LongTensor(data=[label])
            data, sr = librosa.load(p,sr=16000)
            data_tensor = torch.from_numpy(data.T)
            data_tensor = torch.unsqueeze(data_tensor, 0)
            self.datalist.append((data_tensor, label))
            hours += data_tensor.size(1)/16000
        print('IEMOCAP load ', hours/3600, 'hours')
    
    def __getitem__(self, index):
        if self.in_memory:
            (data_tensor,label) = self.datalist[index]
            if data_tensor.size(1) > self.max_len:
                start = random.randint(0, data_tensor.size(1) - self.max_len - 1)
                return (data_tensor[0,start:].unsqueeze(0), label)
            else:
                return (data_tensor, label)
        else:
            (p,label) = self.data_name_list[index]
            label = torch.LongTensor(data=[label])
            data, sr = librosa.load(p,sr=16000)
            data_tensor = torch.from_numpy(data.T)
            data_tensor = torch.unsqueeze(data_tensor, 0)
            if data_tensor.size(1) > self.max_len:
                start = random.randint(0, data_tensor.size(1) - self.max_len - 1)
                return (data_tensor[0,start:].unsqueeze(0), label)
            else:
                return (data_tensor, label)

    def __len__(self):
        if self.in_memory:
            return len(self.datalist)
        else:
            return len(self.load_list)

class EnsembleDataset(Dataset):
    def __init__(self, data_path, index_path, label_list: dict, return_domain=True, dataset_type='train',pre_load=True, in_memory=True,max_len_s=4, debug_flag=False):
        super(EnsembleDataset, self).__init__()
        # init
        self.RAVDESS = RAVDESSDataset(data_path['RAV'], label_list, dataset_type=dataset_type, \
            pre_load=pre_load, debug_flag=debug_flag)
        #self.CREMAD = CREMADDataset(data_path['CRE'], label_list, dataset_type=dataset_type, pre_load=pre_load)
        self.IEMOCAP = IEMOCAPDataset(data_path['IEM'], index_path, label_list, dataset_type=dataset_type, \
            in_memory=in_memory, max_len_s=max_len_s, debug_flag=debug_flag)
        # register
        self.datasets = [self.RAVDESS, self.IEMOCAP]
        # auto calculate Index
        self.index = [len(dataset) for dataset in self.datasets]
        for i in range(1,len(self.index)):
            self.index[i] += self.index[i-1]
        


    def __getitem__(self, in_ind):
        assert(in_ind < len(self))
        for i, ind in enumerate(self.index):
            if in_ind < ind:
                x_ind = in_ind if i == 0 else in_ind - self.index[i-1]
                assert(i < len(self.datasets) and i >= 0)
                return (torch.LongTensor(data=[i]), self.datasets[i][x_ind])

    def __len__(self):
        leng = 0
        for dataset in self.datasets:
            leng += len(dataset)
        return leng

    def get_class_weight(self):
        w = [1/len(dataset) for dataset in self.datasets]
        w = torch.Tensor(data=w)
        return w

# test
if __name__ == "__main__":
    # output:
    # RAVDESS: 4 4 4 4 3 3 3 3 3 3
    # CREMA-D: 0 1 2 3 4 5 0 0 0 1
    label_list = {'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
    path1 = r"F:\Project\FacialAnimation\dataset_cache\RAVDESS"
    dataset1 = RAVDESSDataset(path1, label_list)
    print('RAVDESS')
    for i in range(10):
        data,label = dataset1[i]
        print('idx:', i, 'data:', data.size(), ' label:', label)
    path2 = r'F:\Project\FacialAnimation\dataset_cache\CREMA-D'
    dataset2 = CREMADDataset(path2, label_list)
    print('CREMA-D')
    for i in range(10):
        data,label = dataset2[i]
        print('idx:', i, 'data:', data.size(), ' label:', label)
    
