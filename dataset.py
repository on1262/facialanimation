import yaml, json
import os
import os.path as path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils.config_loader import GBL_CONF, PATH
from utils.converter import (audio2tensor, convert_img, video2sequence, video2wav)

from utils.detector import FANDetector
from fitting.fit_utils import Mesh, read_vl
from utils.interface import DANModel, EMOCAModel



def get_emo_label_from_name(dataset_name, name):
    '''get emotion label from name (e.g. NEU from 1073_TIE_NEU_XX)'''
    if dataset_name == 'cremad':
        n_dict = GBL_CONF['dataset']['cremad']['emo_label']
        for label in n_dict.keys():
            if label in name:
                return n_dict[label]
        assert(0)
    else:
        assert(0)

def zero_padding(t:torch.tensor, pad_size, dim=-1):
    assert(t.size(dim) <= pad_size)
    dev = t.device
    if t.size(dim) == pad_size:
        return t
    if t.dim() == 1:
        return torch.cat([t, torch.zeros((pad_size-t.size(0)), device=dev)])
    else:
        p_size = list(t.size())
        p_size[dim] = pad_size - t.size(dim)
        p_size = torch.Size(p_size)
        return torch.cat([t, torch.zeros(p_size, device=dev)], dim=dim)

def adjust_frame_rate(result_dict, in_fps):
    if in_fps == 30:
        return result_dict
    # all video data should be interpolated to match 30 fps
    itp_len = round(result_dict['params'].size(0) * (30.0/in_fps))
    result_dict['params'] = F.interpolate(result_dict['params'][None,None,...], (itp_len, result_dict['params'].size(1)))[0,0,...]
    result_dict['code_dict'] = {key: F.interpolate(code[None,None,...], \
        (itp_len, code.size(1)) if code.dim() == 2 else (itp_len, code.size(1), code.size(2)))[0,0,...] \
        for key, code in result_dict['code_dict'].items()}

    if 'emo_tensor' in result_dict.keys():
        result_dict['emo_tensor'] = F.interpolate(result_dict['emo_tensor'][None,None,...], (itp_len, result_dict['emo_tensor'].size(1)))[0,0,...]
    if 'imgs' in result_dict.keys():
        # b,3,h,w->h,w,b,3
        assert(result_dict['imgs'].size(1) == 3) # store format
        result_dict['imgs'] = result_dict['imgs'].permute(2,3,0,1)
        result_dict['imgs'] = F.interpolate(result_dict['imgs'], (itp_len, 3))
        result_dict['imgs'] = result_dict['imgs'].permute(2,3,0,1)
    return result_dict

def parse_dataset(dataset, save_path, max_size=-1):
    out_list = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        if data is not None:
            out_list.append(data)
        if len(out_list) % 5 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('parsing: ', len(out_list), ' of ', len(dataset), ' current time=', current_time)
        if len(out_list) > max_size and max_size > 0:
            print('break at size', len(out_list))
            break
    torch.save(out_list, save_path)
    print('parsing dataset: Done')
    return out_list
    
def FACollate_fn(data_list:list, clip_max=True):
    """
        name(list(str))
        emo_label(tensor, size:Batch)
        wav(tensor, size: Batch, max_wav_len)
        imgs: list, size: *, seq_len,3,224,224
        seqs_len: tensor, size: Batch
        code_dict: list(dict(tensor))
    """
    stack_key = {'wav', 'params', 'seqs_len'} #{'name', 'imgs', 'domain', 'code_dict', 'emo_tensor'}
    result_dict = {key:[] for key in data_list[0].keys() }
    
    if clip_max:
        max_seq_length = 120
    else:
        max_seq_length = 10000
    max_wav_length = round(16000/30*max_seq_length)

    # list(dict) -> dict(list)
    for idx in range(len(data_list)):
        data = data_list[idx]
        for key in data.keys():
            #print('key:', key, data[key])
            # TODO emo label is incorrect if using both lrs2 and cremad
            if key in result_dict.keys():
                result_dict[key].append(data[key])

    if 'imgs' in data_list[0].keys():
        seqs_max = max([torch.LongTensor(data=[seq.size(0)]).squeeze() for seq in result_dict['imgs']])
        if seqs_max > 2*max_seq_length:
            print('warning: too large seqs: ', seqs_max)
        if seqs_max > max_seq_length:
            result_dict['imgs'] = [seq[:min(seq.size(0), max_seq_length),...] for seq in result_dict['imgs']]
        # add sequence original length
        result_dict['seqs_len'] = [torch.LongTensor(data=[seq.size(0)]) for seq in result_dict['imgs']]
    elif 'params' in data_list[0].keys(): # TODO not tested
        seqs_max = max([torch.LongTensor(data=[seq.size(0)]).squeeze() for seq in result_dict['params']])
        if seqs_max > 2*max_seq_length:
            print('warning: too large seqs: ', seqs_max)
        result_dict['seqs_len'] = [torch.LongTensor(data=[min(seq.size(0), max_seq_length)]) for seq in result_dict['params']]
    else: # no video input
        result_dict['seqs_len'] = [torch.LongTensor(data=[round(wav.size(-1)*30/16000)]) for wav in result_dict['wav']]

    if 'code_dict' in data_list[0].keys():
        result_dict['code_dict'] = \
            [{key:val[:result_dict['seqs_len'][idx],...] for key,val in codedict.items()} for idx, codedict in enumerate(result_dict['code_dict'])]
    else:
        default_lightcode = torch.Tensor(data=GBL_CONF['inference']['emoca']['default_light_code'])

        # generate default code_dict
        result_dict['code_dict'] = \
            [{'cam':torch.zeros((result_dict['seqs_len'][idx],3)) + torch.Tensor([10,0,0]), \
                'shapecode':torch.zeros((result_dict['seqs_len'][idx],100)), \
                    'expcode':torch.zeros(result_dict['seqs_len'][idx],50), \
                        'lightcode':default_lightcode[None,...].expand(result_dict['seqs_len'][idx],-1,-1), \
                        'posecode':torch.zeros(result_dict['seqs_len'][idx],6)} for idx in range(len(data_list))]

    if 'wav' in data_list[0].keys():
        # padding wav
        wav_max = max([wav.size(-1) for wav in result_dict['wav']])
        s_wav_max = max(result_dict['seqs_len']).item() * round(16000/30)
        result_dict['wav'] = [zero_padding(wav, wav_max, dim=-1)[:min(wav_max, max_wav_length, s_wav_max),...] for wav in result_dict['wav']]

    if 'params' in data_list[0].keys():
        # padding seq
        seqs_max = max([p.size(-2) for p in result_dict['params']])
        s_seq_max = max(result_dict['seqs_len']).item()
        result_dict['params'] = [zero_padding(seq, seqs_max, dim=-2)[:min(s_seq_max, max_seq_length),...] for seq in result_dict['params']]

    if 'emo_tensor' in data_list[0].keys():
        result_dict['emo_tensor'] = [et[:result_dict['seqs_len'][idx]] for idx, et in enumerate(result_dict['emo_tensor'])]
        result_dict['emo_tensor'] = torch.cat(result_dict['emo_tensor'], dim=0)

    for key in result_dict.keys():
        if key in stack_key:
            result_dict[key] = torch.stack(result_dict[key], dim=0)
            if key == 'seqs_len' and result_dict[key].dim() > 1:
                result_dict[key] = result_dict[key].view(-1)
    #print(result_dict['code_dict'][0]['shapecode'].size())
    return result_dict


class CREMADDataset(Dataset):
    """
    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self,
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        emoca=None,
        dan=None,
        debug=0, 
        debug_max_load=32
        ):

        super().__init__()
        self.dataset_name = 'cremad'
        cremad_conf = GBL_CONF['dataset']['cremad']
        self.debug = debug
        self.label_dict = cremad_conf['emo_label']
        self.fps = cremad_conf['fps'] # the actual fps is a bit lower than 30, but it does not matter
        self.data_path = PATH['dataset']['cremad']
        self.cache_path = PATH['dataset']['cache']
        self.dict = cremad_conf['emo_label']
        self.bad_data = cremad_conf['bad_sample']
        self.preloaded_flag = path.exists(path.join(self.cache_path, 'cremad_' + dataset_type + '.pt')) and debug == 0

        if not self.preloaded_flag: # dataset preprocessing
            self.fan = FANDetector(device=device)
            self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
            self.dan = dan if dan is not None else DANModel(device=device)
            # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
            assert(self.fan.device == self.emoca.device == self.dan.device) 
            print('CREMAD: loading from file')
            self.data_list = []
            video_set = {name.split('.')[0] for name in os.listdir(path.join(self.data_path, 'VideoFlash'))} # frames/ included
            for file in sorted(os.listdir(path.join(self.data_path, 'AudioWAV'))): # stable index for every call
                name = file.split('.')[0]
                label = self.label_dict.get(self.dict.get(name.split('_')[2]))
                if (label is not None) and (name in video_set) and (name not in self.bad_data): # avoid bad data
                    self.data_list.append({'name':name, 'emo_label':torch.LongTensor(data=[label]).squeeze()})
                    if debug > 0 and len(self.data_list) >= debug_max_load:
                        break
            if debug == 0: # auto parsing is disabled in debug mode
                # train, test or all
                division = cremad_conf['division']
                tr_end = round(division[0]*len(self.data_list))
                val_end = round((division[0]+division[1])*len(self.data_list))
                if dataset_type == 'train':
                    self.data_list = self.data_list[:tr_end]
                elif dataset_type == 'valid':
                    self.data_list = self.data_list[tr_end:val_end]
                elif dataset_type == 'test':
                    self.data_list = self.data_list[val_end:]
                else:
                    assert(0)
                # auto parse dataset
                print('auto parse dataset')
                self.data_list = parse_dataset(self, path.join(self.data_path, 'cremad_' + dataset_type + '.pt'))
                self.preloaded_flag = True
        else:
            print('CREMAD: loading from cremad_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(self.data_path, 'cremad_' + dataset_type + '.pt'), map_location='cpu')
        
        
        print('CREAMD: load ', len(self.data_list), ' samples')
    
    
    def __getitem__(self, index):
        """
        output:
            result_dict: dict(clone)
                'name': str
                'imgs': cropped, seq_len*3*224*224, type='store'
                'code_dict': dict, key={'shapecode','expcode','pose','lightcode','cam',...}, value=tensor, size=seq_len*code
                'params': tensor, seq_len*56
        """
        if not isinstance(index, int):
            return [self[idx] for idx in index]
        
        result_dict = {}
        ori_dict = self.data_list[index]
        #print('key:', list(ori_dict.keys()))
        # avoid cuda OOM by forcing origin tensors stay in cpu memory. result_dict should be a new dict
        for key in ori_dict.keys():
            if isinstance(ori_dict[key], torch.Tensor):
                result_dict[key] = ori_dict[key].detach().clone()
            else:
                # code_dict, name
                if isinstance(ori_dict[key], dict):
                    result_dict[key] = {k:v.detach().clone() for k,v in ori_dict[key].items()}
                else:
                    result_dict[key] = ori_dict[key]
        
        if not self.preloaded_flag:
            result_dict['wav'] = audio2tensor(path.join(self.data_path, self.aud_dir, result_dict['name']+'.wav')) # load wav
            #load and crop video
            imgs_list = video2sequence(path.join(self.data_path, self.vid_dir, result_dict['name']+'.flv'), return_path=False, o_code='fan_in')

            with torch.no_grad():
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs_list)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return None
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                # generate codedict(for output norm)
                result_dict['code_dict'] = \
                    self.emoca.encode(convert_img(imgs, 'store','emoca').to(self.emoca.device), return_img=False) # dict(seq_len, code)
                for key in result_dict['code_dict'].keys():
                    result_dict['code_dict'][key] = result_dict['code_dict'][key].detach().to('cpu')
                result_dict['params'] = torch.cat([result_dict['code_dict']['expcode'], result_dict['code_dict']['posecode']], dim=-1)
                
        return result_dict
    
    def __len__(self):
        return len(self.data_list)

class BIWIDataset(Dataset):
    """
    @deprecated

    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self,
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        debug=0,
        debug_max_load=32
        ):

        super().__init__()
        self.dataset_name='biwi'
        biwi_conf = GBL_CONF['dataset']['biwi']
        self.debug = debug
        self.fps = biwi_conf['fps']
        self.data_path = PATH['dataset']['biwi']
        self.bad_sample = biwi_conf['bad_sample']
        self.train_subjects = biwi_conf['train_subjects']
        self.test_subjects = biwi_conf['test_subjects']
    
        self.preloaded = path.exists(path.join(self.data_path, 'biwi_' + dataset_type + '.pt')) and debug == 0
    
        if not self.preloaded:
            self.fan = FANDetector(device=device)
            self.dan = DANModel(device=device)
            print('BIWI: loading from file.')
            self.data_list = []
            audio_set = set()
            for name in os.listdir(path.join(self.data_path, 'audio')):
                if 'cut' in name:
                    audio_set.add(name.split('_cut.wav')[0])
            
            subjects = self.train_subjects if dataset_type == 'train' else self.test_subjects
            for person in sorted(os.listdir(path.join(self.data_path, 'images'))): # stable index for every call
                for idx_str in sorted(os.listdir(path.join(self.data_path, 'images', person))):
                    name = person + '_' + idx_str
                    if (person in subjects) and (name in audio_set) and (name not in self.bad_sample):
                        self.data_list.append({'name':name, 
                            'img_path': path.join(self.data_path, 'images', person, idx_str),
                            'params_path': path.join(self.data_path, 'fit_output', person, idx_str),
                            'wav_path': path.join(self.data_path, 'audio', f'{name}_cut.wav')})
                        if debug > 0 and len(self.data_list) >= debug_max_load:
                            break
            if debug == 0: # data already split in _train/test.pt
                # auto parse dataset
                print('auto parse dataset')
                self.data_list = parse_dataset(self, path.join(self.data_path, 'biwi_' + dataset_type + '.pt'))
                self.preloaded = True
        else:
            print('BIWI: loading from biwi_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(self.data_path, 'biwi_' + dataset_type + '.pt'), map_location='cpu')
        
        print('BIWI: load ', len(self.data_list), ' samples')
    
    """
    output:
        result_dict: dict(clone)
            'name': str
            'imgs': cropped, seq_len*3*224*224, type='store'
            'code_dict': dict, key={'shapecode','expcode','pose','lightcode','cam',...}, value=tensor, size=seq_len*code
            'params': tensor, seq_len*56
    """
    def __getitem__(self, index):
        if not isinstance(index, int):
            return [self[idx] for idx in index]
        
        result_dict = {}
        ori_dict = self.data_list[index]
        #print('key:', list(ori_dict.keys()))
        # avoid cuda OOM by forcing origin tensors stay in cpu memory. result_dict should also be a new dict
        for key in ori_dict.keys():
            if isinstance(ori_dict[key], torch.Tensor):
                result_dict[key] = ori_dict[key].detach().clone()
            else:
                # code_dict, name
                if isinstance(ori_dict[key], dict):
                    result_dict[key] = {k:v.detach().clone() for k,v in ori_dict[key].items()}
                else:
                    result_dict[key] = ori_dict[key]
        
        if not self.preloaded:
            #load wav
            result_dict['wav'] = audio2tensor(result_dict['wav_path'])
            #load and crop video
            imgs_list = [read_image(os.path.join(result_dict['img_path'], p)) for p in sorted(os.listdir(result_dict['img_path']))]
            imgs = convert_img(torch.stack(imgs_list, dim=0), 'tv', 'fan_in')
            with torch.no_grad():
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return None
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                # result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan')).cpu()
                # load params from json
                with open(path.join(result_dict['params_path'], 'params.json'), 'r') as fp:
                    params = torch.as_tensor(json.load(fp)) # seq, 156
                    result_dict['params'] = params[:,100:156]
                    result_dict['code_dict'] = {'shapecode': params[:,:100], 'expcode':params[:,100:150], 'posecode':params[:,150:]}
                
        result_dict['emo_tensor'] = torch.zeros((result_dict['params'].size(0), 7)) # biwi emo-tensor is not reliable
        return result_dict
    
    def __len__(self):
        return len(self.data_list)


class BaselineBIWIDataset(BIWIDataset):
    '''
    @deprecated

    biwi dataset(test mode) + directly load .vl files
    '''
    def __init__(self, data_path, device):
        super().__init__(data_path, label_dict=None, dataset_type='test', device=device, debug=0)    

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        person, idx_str = result_dict['name'].split('_')
        vl_dir = os.path.join(self.data_path, 'fit', person, idx_str)
        flame_template = os.path.join(self.data_path, 'fit_output', person, idx_str,'params.json')
        verts_list = []
        for vl_file in sorted(os.listdir(vl_dir)): # 001.vl
            vl_arr = torch.from_numpy(read_vl(os.path.join(vl_dir, vl_file)))
            verts_list.append(vl_arr)
        with open(flame_template, 'r') as f:
            params = torch.as_tensor(json.load(f)) # seqs_len*156
            result_dict['flame_template'] = params[0,0:100].unsqueeze(0).expand(params.size(0),-1)
        result_dict['verts'] = torch.stack(verts_list, dim=0)
        result_dict['seqs_len'] = torch.LongTensor(data=[len(verts_list)])
        return result_dict


class VOCASET(Dataset):
    """
    VOCASET 60fps, 22kHz
    """
    def __init__(self, 
        dataset_type='train', 
        debug=0, 
        debug_max_load=16
        ):
        super().__init__()
        self.dataset_name = 'vocaset'
        vocaset_conf = GBL_CONF['dataset']['vocaset']
        self.debug = debug
        self.data_path = PATH['dataset']['vocaset'] # .../VOCASET
        self.dataset_type = dataset_type
        self.bad_data = {}
        self.emo_only = vocaset_conf['emo_only'] # only use emotion sentences
        self.subjects = {
            'train': vocaset_conf['train_subjects'],
            'valid': vocaset_conf['valid_subjects'],
            'test': vocaset_conf['test_subjects']
        }
        self.min_sec, self.max_sec = vocaset_conf['min_sec'], vocaset_conf['max_sec']
        self.fps = vocaset_conf['fps']
        self.preloaded = path.exists(path.join(self.data_path, 'vocaset_' + dataset_type + '.pt')) and debug == 0
        if not self.preloaded:
            print('VOCASET: loading from file')
            assert(dataset_type != 'all') # not supported yet
            self.data_list = []
            exit_flag = False
            subjects = self.subjects[dataset_type]
            for person in sorted(os.listdir(self.data_path)):
                if exit_flag:
                    break
                if 'FaceTalk_' in person and person in subjects:
                    ft_path = os.path.join(self.data_path, 'fit_output', person)
                    for sentence in sorted(os.listdir(ft_path)): # sentenceXX
                        if self.emo_only and int(sentence.split('ce')[-1]) <= 20: # sentence 1-20 are not emotional
                            continue
                        name_code = person + '='+ sentence # FaceTalk_XX_sentenceYY
                        self.data_list.append({'name':name_code, 
                            'path': os.path.join(ft_path, sentence), 
                            'wav_path':  os.path.join(self.data_path, 'audio', person, sentence+'.wav')}) # name_code and real path
                        if debug > 0 and len(self.data_list) >= debug_max_load:
                            exit_flag = True
                            break
            self.data_list = sorted(self.data_list, key=lambda data: data['name'])
            if debug == 0:
                # auto parse dataset
                print('Auto parsing dataset')
                self.data_list = parse_dataset(self, path.join(self.data_path, 'vocaset_' + dataset_type + '.pt'))
                self.preloaded = True
        else:
            print('VOCASET: loading from vocaset_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(self.data_path, 'vocaset_' + dataset_type + '.pt'), map_location='cpu')

        print('VOCASET: load ', len(self.data_list), ' samples')
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            return [self[idx] for idx in index]
        
        result_dict = {}
        ori_dict = self.data_list[index]
        # print('key:', list(ori_dict.keys()))
        # avoid cuda OOM by forcing origin tensors stay in cpu memory. result_dict should also be a new dict
        for key in ori_dict.keys():
            if isinstance(ori_dict[key], torch.Tensor):
                result_dict[key] = ori_dict[key].detach().clone()
            else:
                # code_dict, name
                if isinstance(ori_dict[key], dict):
                    result_dict[key] = {k:v.detach().clone() for k,v in ori_dict[key].items()}
                else:
                    result_dict[key] = ori_dict[key]
        
        if not self.preloaded:
            # load wav
            # print('wav path:', result_dict['wav_path'])
            result_dict['wav'] = audio2tensor(result_dict['wav_path'])
            if result_dict['wav'].size(0) < 16000*self.min_sec or result_dict['wav'].size(0) > 16000*self.max_sec:
                return None
            
            # read fitted samples (code dict generated by fitting module)
            with open(os.path.join(result_dict['path'], 'params.json'), 'r') as fp:
                params = torch.asarray(json.load(fp))
            result_dict['params'] = params[:, 100:]
            result_dict['code_dict'] = {'shapecode': params[:,:100], 'expcode':params[:,100:150], 'posecode':params[:,150:]}
            result_dict = adjust_frame_rate(result_dict, self.fps)
            result_dict['emo_tensor'] = torch.zeros((result_dict['params'].size(0), 7))

        return result_dict
        
    def __len__(self):
        return len(self.data_list)



class BaselineVOCADataset(VOCASET):
    '''
    Directly load obj files
    '''
    def __init__(self, dataset_type, device):
        super().__init__(dataset_type=dataset_type, device=device, debug=0)
        self.template = {}
        self.person_path = os.path.join(self.data_path, 'fit_output')
        for person in sorted(os.listdir(self.person_path)):
            self.template[person] = {
                'obj': Mesh(Mesh.read_obj(os.path.join(self.person_path, person, 'sentence01','ori', '0.obj')),'flame'),
                'ply': os.path.join(self.data_path, 'templates',  person + '.ply')
            }

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        person, idx_str = result_dict['name'].split('=')
        obj_dir = os.path.join(self.data_path, 'fit_output', person, idx_str)
        verts_list = []
        for idx in range(len(os.listdir(os.path.join(obj_dir, 'ori')))):
            obj_path = os.path.join(obj_dir, 'ori', f'{idx}.obj') # load origin data
            verts_list.append(torch.as_tensor(Mesh.read_obj(obj_path)))
        result_dict['verts'] = torch.stack(verts_list, dim=0) # seq_len, 5023, 3
        # adjust frame rate to 30
        itp_len = round(len(verts_list) * 0.5)
        params = result_dict['params']
        try:
            assert(params.size(0) == itp_len)
        except:
            print('error: params', params.size(0), 'ipt:', itp_len)
            min_len = min(itp_len, params.size(0))
            itp_len = min_len
            result_dict['params'] = params[:min_len,:]
            result_dict['codedict'] = {k:v[:min_len,...] for k,v in result_dict['codedict'].items()}
        
        result_dict['flame_template'] = self.template[person]
        result_dict['shapecode'] = result_dict['code_dict']['shapecode']
        result_dict['wav'] = result_dict['wav'].unsqueeze(0)
        assert(result_dict['wav'].dim() == 2)
        result_dict['verts'] = F.interpolate(result_dict['verts'][None,None,...], (itp_len, result_dict['verts'].size(1),3))[0,0,...]
        result_dict['seqs_len'] = torch.LongTensor(data=[itp_len])
        return result_dict



class LRS2Dataset(Dataset):
    """
    LRS2 dataset
    able to parse, but too large(more than 100000) and low quality
    use 2000/3000 to enlarge sentence diversity
    """
    
    def __init__(self, 
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        emoca=None, 
        debug=0, 
        debug_max_load=160
        ):
        """
        data(dict):
            keys: name, emo_label, wav, imgs
        """
        super().__init__()
        self.dataset_name = 'lrs2'
        self.debug = debug
        lrs2_conf = GBL_CONF['dataset']['lrs2']
        self.data_path = PATH['dataset']['lrs2'] # .../LRS2
        self.bad_sample = lrs2_conf['bad_sample']
        self.min_sec = lrs2_conf['min_sec']
        self.max_sec = lrs2_conf['max_sec']
        self.fps = lrs2_conf['fps']
        self.preloaded = path.exists(path.join(self.data_path, 'lrs2_' + dataset_type + '.pt')) and debug == 0

        if not self.preloaded:
            self.fan = FANDetector(device=device)
            self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
            self.dan = DANModel(device=device)
            assert(self.fan.device == self.emoca.device) # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
            print('LRS2: loading from file')
            txt_name = 'train.txt' if dataset_type == 'train' else 'val.txt'
            assert(dataset_type != 'all') # not supported yet
            with open(os.path.join(self.data_path,txt_name)) as file:
                lines = file.readlines() # e.g. 6300370419826092098/00007
                lines = [line.rstrip() for line in lines]
                self.video_set = set(lines)
            self.data_list = []
            exit_flag = False
            for root, dirs, files in os.walk(self.data_path):
                if exit_flag:
                    break
                for name in files:
                    if name.endswith('.mp4'):
                        name_code = os.path.split(root)[-1] + '/' + name.split('.')[0]
                        #print(name_code)
                        if name_code in self.video_set:
                            self.data_list.append({'name':name_code, 'path': os.path.join(root, name)}) # name_code and real path
                            if debug > 0 and len(self.data_list) >= 16:
                                exit_flag = True
                                break
            self.data_list = sorted(self.data_list, key=lambda data: data['name'])

            if debug == 0: # data already split in _train/test.pt
                # auto parse dataset
                print('auto parse dataset')
                self.data_list = parse_dataset(self, path.join(self.data_path, 'lrs2_' + dataset_type + '.pt'), max_size=3000 if dataset_type == 'train' else 500)
                self.preloaded = True
        else:
            print('LRS2: loading from lrs2_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(self.data_path, 'lrs2_' + dataset_type + '.pt'), map_location='cpu')
        
        
        print('LRS2: load ', len(self.data_list), ' samples')
    
    
    def __getitem__(self, index):
        """
        output:
            result_dict: dict(clone)
                'name': str
                'imgs': cropped, seq_len*3*224*224, type='store'
                'code_dict': dict, key={'shapecode','expcode','pose','lightcode','cam',...}, value=tensor, size=seq_len*code
                'params': tensor, seq_len*56
        """
        if not isinstance(index, int):
            return [self[idx] for idx in index]
        
        result_dict = {}
        ori_dict = self.data_list[index]
        #print('key:', list(ori_dict.keys()))
        # avoid cuda OOM by forcing origin tensors stay in cpu memory. result_dict should also be a new dict
        for key in ori_dict.keys():
            if isinstance(ori_dict[key], torch.Tensor):
                result_dict[key] = ori_dict[key].detach().clone()
            else:
                # code_dict, name
                if isinstance(ori_dict[key], dict):
                    result_dict[key] = {k:v.detach().clone() for k,v in ori_dict[key].items()}
                else:
                    result_dict[key] = ori_dict[key]
        
        if not self.preloaded:
            #load wav
            result_dict['wav'] = audio2tensor(video2wav(result_dict['path'], os.path.join(self.data_path, 'cache')))
            if result_dict['wav'].size(0) < 16000*self.min_sec or result_dict['wav'].size(0) > 16000*self.max_sec:
                return None
            #load and crop video
            imgs_list = video2sequence(result_dict['path'], return_path=False, o_code='fan_in')
            #imgs = torch.stack(imgs_list, dim=0)
            with torch.no_grad():
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs_list)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return None
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                #generate codedict(for output norm)
                result_dict['code_dict'] = \
                    self.emoca.encode(convert_img(imgs, 'store','emoca').to(self.emoca.device), return_img=False) # dict(seq_len, code)
                for key in result_dict['code_dict'].keys():
                    result_dict['code_dict'][key] = result_dict['code_dict'][key].detach().to('cpu')
                result_dict['params'] = torch.cat([result_dict['code_dict']['expcode'], result_dict['code_dict']['posecode']], dim=-1)
                result_dict.pop('path')
            result_dict = adjust_frame_rate(result_dict, self.fps)
            
        return result_dict
    
    def __len__(self):
        return len(self.data_list)


class EnsembleDataset(Dataset):
    def __init__(self, dataset_type='train', device=torch.device('cuda:0'), return_domain=False, emoca=None, dan=None, debug=0):
        super(EnsembleDataset, self).__init__()
        self.return_domain = return_domain
        self.dataset_type = dataset_type
        self.LRS2 = LRS2Dataset(dataset_type=dataset_type, device=device, emoca=emoca, debug=debug)
        self.CREMAD = CREMADDataset(dataset_type=dataset_type, device=device, emoca=emoca, dan=dan, debug=debug)
        #self.BIWI = BIWIDataset(data_path['BIWI'], label_dict, dataset_type=dataset_type, device=device, debug=debug)
        self.VOCASET = VOCASET(dataset_type=dataset_type, device=device, debug=debug)
        
        self.datasets = {'cremad': self.CREMAD, 'vocaset': self.VOCASET, 'lrs2': self.LRS2}
        self.datasets = [self.datasets[key] for key in GBL_CONF['dataset']['ensemble_dataset']['datasets']]
        # calculate Index
        self.index = []
        for idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((idx, sample_idx))

        for idx, dataset in enumerate(self.datasets):
            print(dataset_type, ' examples in datasets[', idx, '] ', [data['name'] for data in dataset[range(min(20, len(dataset)))]]) # output first 20 data
        
        print('Ensemble dataset: done')
    
    def check_params_distribution(self):
        # check distribution
        print('Ensemble Dataset: check distribution')
        distri = {}
        for dataset in self.datasets:
            dist = torch.zeros(56)
            for data in dataset:
                params = data['params']
                dist += torch.mean(torch.abs(params), dim=0)
            dist /= len(dataset)
            distri[dataset.dataset_name] = dist.tolist()
        with open(os.path.join(PATH['dataset']['cache'], 'distribution.yml'), 'w') as f:
            yaml.dump(distri, f)

    
    def __getitem__(self, in_ind):
        assert(in_ind < len(self))
        dataset_idx, sample_idx = self.index[in_ind]
        result_dict = self.datasets[dataset_idx][sample_idx]
        domain = self.datasets[dataset_idx].dataset_name
        if self.return_domain:
            result_dict.update({'domain': domain})
        # generate emo_tensor_conf
        if domain in ['lrs2', 'cremad']:
            result_dict['emo_tensor_conf'] = 'use'
        else:
            result_dict['emo_tensor_conf'] = 'no_use'
        return result_dict

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def get_class_weight(self):
        w = [1/len(dataset) for dataset in self.datasets]
        w = torch.Tensor(data=w)
        return w


class TESTDataset(Dataset):
    """
    Generate a Dataset from inference input

    Available files:
        - audio: .wav
        - video: .mp4, .flv
    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self, 
        device=torch.device('cuda:0'), 
        emoca=None,
        dan=None
        ):
        super().__init__()
        conf = GBL_CONF['inference']['infer_dataset']
        self.in_dir = PATH['inference']['input']
        self.out_dir = PATH['inference']['output']
        self.cache_dir = PATH['inference']['cache']
        self.emo_set = conf['emo_set']
        self.fan = FANDetector(device=device)
        self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
        self.dan = dan if dan is not None else DANModel(device=device)
        assert(self.fan.device == self.emoca.device) # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
        print('TEST: loading from file')
        self.data_list = []
        for file in sorted(os.listdir(self.in_dir)): # stable index for every call
            name = file.split('.')[0]
            if file.split('.')[-1] not in ['mp4', 'flv', 'wav']:
                continue
            # try fetch emo label
            label = None
            for emo_label in self.emo_set:
                if emo_label in name:
                    label = emo_label
            if label is not None: # avoid bad data
                self.data_list.append({'name':name, 'emo_label':torch.LongTensor(data=[label]).squeeze(), 'path':os.path.join(self.in_dir, file)})
            else:
                self.data_list.append({'name':name, 'path':os.path.join(self.in_dir, file)})
        
        print('TEST: load ', len(self.data_list), ' samples')
    
    def __getitem__(self, index):
        """
        output:
            result_dict: dict(clone)
                'name': str
                'imgs': cropped, seq_len*3*224*224, type='store'
                'code_dict': dict, key={'shapecode','expcode','pose','lightcode','cam',...}, value=tensor, size=seq_len*code
                'params': tensor, seq_len*56
        """
        if not isinstance(index, int):
            return [self[idx] for idx in index]
        
        result_dict = {}
        ori_dict = self.data_list[index]
        # avoid cuda OOM by forcing origin tensors stay in cpu memory. result_dict should also be a new dict
        for key in ori_dict.keys():
            if isinstance(ori_dict[key], torch.Tensor):
                result_dict[key] = ori_dict[key].detach().clone()
            else:
                # code_dict, name
                if isinstance(ori_dict[key], dict):
                    result_dict[key] = {k:v.detach().clone() for k,v in ori_dict[key].items()}
                else:
                    result_dict[key] = ori_dict[key]
        
        #load wav
        if result_dict['path'].endswith('.flv') or result_dict['path'].endswith('.mp4'):
            result_dict['wav_path'] = video2wav(result_dict['path'], self.cache_dir) # used to load wav for final output
            result_dict['wav'] = audio2tensor(result_dict['wav_path'])
            #load and crop video
            imgs_list = video2sequence(result_dict['path'], return_path=False, o_code='fan_in')
            #imgs = torch.stack(imgs_list, dim=0)
            with torch.no_grad():
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs_list)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return result_dict
                else:
                    imgs = convert_img(crop_imgs, 'fan_out', 'store')
                    result_dict['imgs'] = imgs
                    result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                    #generate codedict(for output norm)
                    result_dict['code_dict'] = \
                        self.emoca.encode(convert_img(imgs, 'store','emoca').to(self.emoca.device), return_img=False) # dict(seq_len, code)
                    for key in result_dict['code_dict'].keys():
                        result_dict['code_dict'][key] = result_dict['code_dict'][key].detach().to('cpu')
                    # reset position and camera code
                    result_dict['code_dict']['posecode'][:, :3] = 0
                    result_dict['code_dict']['posecode'][:, 4:] = 0
                    result_dict['code_dict']['cam'][:,1] = 0
                    result_dict['code_dict']['cam'][:,2] = 0
                    result_dict['code_dict']['cam'][:,0] = 9 # stablize camera position
                    result_dict['params'] = torch.cat([result_dict['code_dict']['expcode'], result_dict['code_dict']['posecode']], dim=-1)
            # calculate frame rate
            if 'params' in result_dict.keys():
                fps = round(result_dict['params'].size(0) / (result_dict['wav'].size(0) / 16000))
                print('fps estimated is ', fps)
                result_dict = adjust_frame_rate(result_dict, fps)
        elif result_dict['path'].endswith('.wav'):
            result_dict['wav_path'] = result_dict['path']
            result_dict['wav'] = audio2tensor(result_dict['path'])
        else:
            print('error path:', result_dict['path'])
            assert(0)
        return result_dict
    
    def __len__(self):
        return len(self.data_list)
