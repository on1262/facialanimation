from utils.config_loader import GBL_CONF, PATH
from torch.utils.data import Dataset
from utils.converter import video2sequence, audio2tensor, convert_img, video2wav
from utils.detector import FANDetector
from utils.check_distribution import check_params_distribution
from utils.interface import EMOCAModel, DANModel
from utils.fitting.fit_utils import read_vl, Mesh
import torch.nn.functional as F
from torchvision.io import read_image
from datetime import datetime
import torch
import os
from plyfile import PlyData
import os.path as path
import json

def get_emo_label_from_name(dataset_name, label_dict:dict, name):
    if dataset_name =='cremad':
        n_dict = {'ANG':'ANG','DIS':'DIS','FEA':'FEA','HAP':'HAP','NEU':'NEU','SAD':'SAD'}
        label = name.split('_')[2]
        label = label_dict.get(n_dict.get(label))
        return label

def get_emo_index(model_name, label):
    if model_name == 'dan':
        labels = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
        convert_list = {0:0, 1:1, 3:2, 4:5, 5:4, 2:6}
        return convert_list[label], labels[convert_list[label]]

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

def parse_dataset(dataset, save_path, max_size=100000):
    out_list = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        if data is not None:
            out_list.append(data)
        if len(out_list) % 5 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('parsing: ', len(out_list), ' of ', len(dataset), ' current time=', current_time)
        if len(out_list) > max_size:
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
        label_dict:dict, 
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        emoca=None, 
        debug=0, 
        vid_dir='VideoFlash', 
        aud_dir='AudioWAV',
        debug_max_load=32
        ):

        super().__init__()
        self.dataset_name = 'cremad'
        self.debug = debug
        self.vid_dir = vid_dir
        self.aud_dir = aud_dir
        self.fps = 30.0 # a bit lower than 30, but that does not matter
        self.data_path = PATH['dataset']['cremad']
        self.dict = GBL_CONF['dataset']['cremad']['emo_label']
        self.bad_data = GBL_CONF['dataset']['cremad']['bad_sample']
        self.preloaded = path.exists(path.join(self.data_path, 'cremad_' + dataset_type + '.pt')) and debug == 0
        if not self.preloaded: # dataset preprocessing
            self.fan = FANDetector(device=device)
            self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
            self.dan = DANModel(device=device)
            assert(self.fan.device == self.emoca.device) # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
            print('CREMAD: loading from file')
            self.data_list = []
            video_set = {name.split('.')[0] for name in os.listdir(path.join(data_path, vid_dir))} # frames/ included
            for file in sorted(os.listdir(path.join(data_path, aud_dir))): # stable index for every call
                name = file.split('.')[0]
                label = label_dict.get(self.dict.get(name.split('_')[2]))
                if (label is not None) and (name in video_set) and (name not in self.bad_data): # avoid bad data
                    self.data_list.append({'name':name, 'emo_label':torch.LongTensor(data=[label]).squeeze()})
                    if debug > 0 and len(self.data_list) >= debug_max_load:
                        break
            if debug == 0: # data already split in _train/test.pt
                # train, test or all
                offset = round(0.8*len(self.data_list))
                if dataset_type == 'train':
                    self.data_list = self.data_list[:offset]
                elif dataset_type == 'test':
                    self.data_list = self.data_list[offset:]
                # auto parse dataset
                print('auto parse dataset')
                self.data_list = parse_dataset(self, path.join(data_path, 'cremad_' + dataset_type + '.pt'))
                self.preloaded = True
        else:
            print('CREMAD: loading from cremad_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(data_path, 'cremad_' + dataset_type + '.pt'), map_location='cpu')
        
        
        print('CREAMD: load ', len(self.data_list), ' samples')
    
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
            result_dict['wav'] = audio2tensor(path.join(self.data_path, self.aud_dir, result_dict['name']+'.wav'))
            #load and crop video
            imgs_list = video2sequence(path.join(self.data_path, self.vid_dir, result_dict['name']+'.flv'), return_path=False, o_code='fan_in')
            #imgs = torch.stack(imgs_list, dim=0)
            with torch.no_grad():
                #result_dict['imgs'] = convert_img(self.fan.crop(imgs_list), 'fan_out', 'store') # OK
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs_list)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return None
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                #generate codedict(for output norm)
                #print('before encode:', torch.cuda.max_memory_allocated(self.emoca.device) / (1024*1024*1024), ' GB')
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
    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self, 
        data_path, 
        label_dict:dict, 
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        debug=0, 
        vid_dir='images', 
        aud_dir='audio',
        debug_max_load=32
        ):

        super().__init__()
        self.dataset_name='biwi'
        self.debug = debug
        self.vid_dir = vid_dir
        self.aud_dir = aud_dir
        self.fps = 25.0
        self.data_path = data_path # .../BIWI
        self.bad_data = {'F1_36','F1_39','F1_e20' ,'F3_e02','F3_e03','F4_27','F6_36', 'M1_e01', 'M1_e02', 'M1_e03','M1_e04','M1_e05','M1_e06','M1_e07','M1_e08','M1_e09','M1_e10','M1_e11','M1_e12', 'M1_e13', 'M1_e14','M1_e15','M1_e16','M1_e17','M1_e18','M1_e19','M1_e20', 'M2_20', 'M6_21','F1_e12','F6_e21','F6_e31','M2_31','M2_e04','M2_e08','M6_e02','M6_e18','M6_e27'}
        self.train_subjects = {'F1', 'F2', 'F3', 'F4', 'F5', 'M3', 'M4'}
        self.test_subjects = {'F6', 'F7', 'F8', 'M1', 'M2', 'M6'}
        self.preloaded = path.exists(path.join(data_path, 'biwi_' + dataset_type + '.pt')) and debug == 0
        if not self.preloaded:
            self.fan = FANDetector(device=device)
            self.dan = DANModel(device=device)
            print('BIWI: loading from file.')
            self.data_list = []
            audio_set = set()
            for name in os.listdir(path.join(data_path, self.aud_dir)):
                if 'cut' in name:
                    audio_set.add(name.split('_cut.wav')[0])
            
            subjects = self.train_subjects if dataset_type == 'train' else self.test_subjects
            for person in sorted(os.listdir(path.join(data_path, vid_dir))): # stable index for every call
                for idx_str in sorted(os.listdir(path.join(data_path, vid_dir, person))):
                    name = person + '_' + idx_str
                    if (person in subjects) and (name in audio_set) and (name not in self.bad_data):
                        self.data_list.append({'name':name, 
                            'img_path': path.join(data_path, vid_dir, person, idx_str),
                            'params_path': path.join(data_path, 'fit_output', person, idx_str),
                            'wav_path': path.join(data_path, 'audio', f'{name}_cut.wav')})
                        if debug > 0 and len(self.data_list) >= debug_max_load:
                            break
            if debug == 0: # data already split in _train/test.pt
                # auto parse dataset
                print('auto parse dataset')
                self.data_list = parse_dataset(self, path.join(data_path, 'biwi_' + dataset_type + '.pt'))
                self.preloaded = True
        else:
            print('BIWI: loading from biwi_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(data_path, 'biwi_' + dataset_type + '.pt'), map_location='cpu')
        
        
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

'''
biwi dataset(test mode) + directly load .vl files
'''
class BaselineBIWIDataset(BIWIDataset):
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

"""
VOCASET
60fps, 22kHz

"""
class VOCASET(Dataset):
    def __init__(self, 
        data_path, 
        dataset_type='train', 
        device=None, 
        emoca=None, 
        debug=0, 
        debug_max_load=160
        ):
        super().__init__()
        self.dataset_name = 'vocaset'
        self.debug = debug
        self.data_path = data_path # .../VOCASET
        self.dataset_type = dataset_type
        self.bad_data = {}
        self.emo_only = True # only use emotion sentences
        self.train_subjects = {'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA',
            'FaceTalk_170811_03274_TA','FaceTalk_170913_03279_TA' ,'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA','FaceTalk_170811_03275_TA',
            'FaceTalk_170908_03277_TA'}
        self.test_subjects = {'FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA'}
        self.min_sec = 1.0
        self.max_sec = 10.0
        self.fps = 60.0
        self.preloaded = path.exists(path.join(data_path, 'vocaset_' + dataset_type + '.pt')) and debug == 0
        if not self.preloaded:
            print('VOCASET: loading from file')
            assert(dataset_type != 'all') # not supported yet
            self.data_list = []
            exit_flag = False
            subjects = self.train_subjects if dataset_type == 'train' else self.test_subjects
            for person in sorted(os.listdir(self.data_path)):
                if exit_flag:
                    break
                if 'FaceTalk_' in person and person in subjects:
                    ft_path = os.path.join(self.data_path, 'fit_output', person)
                    for sentence in sorted(os.listdir(ft_path)): # sentenceXX
                        if self.emo_only and int(sentence.split('ce')[-1]) <= 20:
                            continue
                        name_code = person + '='+ sentence # FaceTalk_XX_sentenceYY
                        self.data_list.append({'name':name_code, 
                            'path': os.path.join(ft_path, sentence), 
                            'wav_path':  os.path.join(self.data_path, 'audio', person, sentence+'.wav')}) # name_code and real path
                        if debug > 0 and len(self.data_list) >= 16:
                            exit_flag = True
                            break
            self.data_list = sorted(self.data_list, key=lambda data: data['name'])
            if debug == 0: # data already split in _train/test.pt
                # auto parse dataset
                print('Auto parsing dataset')
                self.data_list = parse_dataset(self, path.join(data_path, 'vocaset_' + dataset_type + '.pt'), max_size=3000 if dataset_type == 'train' else 500)
                self.preloaded = True
        else:
            print('VOCASET: loading from vocaset_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(data_path, 'vocaset_' + dataset_type + '.pt'), map_location='cpu')

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
            
            # read fitted files
            with open(os.path.join(result_dict['path'], 'params.json'), 'r') as fp:
                params = torch.asarray(json.load(fp))
            result_dict['params'] = params[:, 100:]
            
            result_dict['code_dict'] = {'shapecode': params[:,:100], 'expcode':params[:,100:150], 'posecode':params[:,150:]}
            
            result_dict = adjust_frame_rate(result_dict, self.fps)
        
            result_dict['emo_tensor'] = torch.zeros((result_dict['params'].size(0), 7))

        return result_dict
        
    def __len__(self):
        return len(self.data_list)


'''
vocaset test mode
directly load obj files
'''
class BaselineVOCADataset(VOCASET):
    def __init__(self, data_path, device):
        super().__init__(data_path, dataset_type='test', device=device, debug=0)
        self.template = {}
        self.person_path = os.path.join(data_path, 'fit_output')
        for person in sorted(os.listdir(self.person_path)):
            self.template[person] = {
                'obj': Mesh(Mesh.read_obj(os.path.join(self.person_path, person, 'sentence01','ori', '0.obj')),'flame'),
                'ply': os.path.join(data_path, 'templates',  person + '.ply')
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


"""
LRS2 dataset
able to parse, but too large(more than 100000) and low quality
use 2000/3000 to enlarge sentence diversity
"""
class LRS2Dataset(Dataset):
    """
    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self, 
        data_path, 
        dataset_type='train', 
        device=torch.device('cuda:0'), 
        emoca=None, 
        debug=0, 
        debug_max_load=160
        ):

        super().__init__()
        self.dataset_name = 'lrs2'
        self.debug = debug
        self.data_path = data_path # .../LRS2
        self.bad_data = {}
        self.min_sec = 1.0
        self.max_sec = 4.0
        self.fps = 25.0
        self.preloaded = path.exists(path.join(data_path, 'lrs2_' + dataset_type + '.pt')) and debug == 0
        if not self.preloaded:
            self.fan = FANDetector(device=device)
            self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
            self.dan = DANModel(device=device)
            assert(self.fan.device == self.emoca.device) # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
            print('LRS2: loading from file')
            txt_name = 'train.txt' if dataset_type == 'train' else 'val.txt'
            assert(dataset_type != 'all') # not supported yet
            with open(os.path.join(data_path,txt_name)) as file:
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
                self.data_list = parse_dataset(self, path.join(data_path, 'lrs2_' + dataset_type + '.pt'), max_size=3000 if dataset_type == 'train' else 500)
                self.preloaded = True
        else:
            print('LRS2: loading from lrs2_' + dataset_type + '.pt')
            self.data_list = torch.load(path.join(data_path, 'lrs2_' + dataset_type + '.pt'), map_location='cpu')
        
        
        print('LRS2: load ', len(self.data_list), ' samples')
    
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
            result_dict['wav'] = audio2tensor(video2wav(result_dict['path'], os.path.join(self.data_path, 'cache')))
            if result_dict['wav'].size(0) < 16000*self.min_sec or result_dict['wav'].size(0) > 16000*self.max_sec:
                return None
            #load and crop video
            imgs_list = video2sequence(result_dict['path'], return_path=False, o_code='fan_in')
            #imgs = torch.stack(imgs_list, dim=0)
            with torch.no_grad():
                #result_dict['imgs'] = convert_img(self.fan.crop(imgs_list), 'fan_out', 'store') # OK
                # save emo tensor and params only
                crop_imgs = self.fan.crop(imgs_list)
                if crop_imgs is None:
                    print('error unable to crop ', result_dict['name'])
                    return None
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                #generate codedict(for output norm)
                #print('before encode:', torch.cuda.max_memory_allocated(self.emoca.device) / (1024*1024*1024), ' GB')
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
    def __init__(self, label_dict: dict, return_domain=False, dataset_type='train', device=torch.device('cuda:0'), emoca=None, debug=0):
        super(EnsembleDataset, self).__init__()
        self.return_domain = return_domain
        self.dataset_type = dataset_type
        self.LRS2 = LRS2Dataset(data_path['LRS2'], dataset_type=dataset_type, device=device, emoca=emoca, debug=debug)
        self.CREMAD = CREMADDataset(data_path['CRE'], label_dict, dataset_type=dataset_type, device=device, emoca=emoca, debug=debug)
        #self.BIWI = BIWIDataset(data_path['BIWI'], label_dict, dataset_type=dataset_type, device=device, debug=debug)
        self.VOCASET = VOCASET(data_path['VOCASET'], dataset_type=dataset_type, device=device, debug=debug)
        # register
        # self.datasets = [self.CREMAD]
        self.datasets = [self.CREMAD, self.VOCASET, self.LRS2]
        # check distribution
        print('Ensemble Dataset: check distribution')
        distri = {}
        for dataset in self.datasets:
            distri[dataset.dataset_name] = check_params_distribution(dataset).tolist()
        with open('distribution.json', 'w') as f:
            json.dump(distri, f)
        print('Distribution dumped at distribution.log')

        # calculate Index
        self.index = []
        for idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((idx, sample_idx))

        for idx, dataset in enumerate(self.datasets):
            print(dataset_type, ' examples in datasets[', idx, '] ', [data['name'] for data in dataset[range(min(20, len(dataset)))]]) # output first 20 data
        
        print('Ensemble dataset: done')

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
    data(dict):
        keys: name, emo_label, wav, imgs
    """
    def __init__(self, 
        data_path, 
        cache_path,
        label_dict:dict, 
        device=torch.device('cuda:8'), 
        emoca=None, 
        ):

        super().__init__()
        self.data_path = data_path # dir/XX.mp4 or XX.flv
        self.cache_path = cache_path
        self.emo_set = {'ANG','DIS','FEA','HAP','NEU','SAD'}
        self.fan = FANDetector(device=device)
        self.emoca = emoca if emoca is not None else EMOCAModel(device=device)
        self.dan = DANModel(device=device)
        assert(self.fan.device == self.emoca.device) # emoca encoder will throw a cuda OOM when two devices are different (debug mode only)
        print('TEST: loading from file')
        self.data_list = []
        for file in sorted(os.listdir(data_path)): # stable index for every call
            name = file.split('.')[0]
            # try fetch emo label
            label = None
            for str in self.emo_set:
                if str in name:
                    label = label_dict.get(str)
            if label is not None: # avoid bad data
                self.data_list.append({'name':name, 'emo_label':torch.LongTensor(data=[label]).squeeze(), 'path':os.path.join(data_path, file)})
            else:
                self.data_list.append({'name':name, 'path':os.path.join(data_path, file)})
        
        print('TEST: load ', len(self.data_list), ' samples')
    
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
        
        #load wav
        if result_dict['path'].endswith('.flv') or result_dict['path'].endswith('.mp4'):
            result_dict['wav_path'] = video2wav(result_dict['path'], self.cache_path) # used to load wav for final output
            result_dict['wav'] = audio2tensor(result_dict['wav_path'])
        elif result_dict['path'].endswith('.wav'):
            result_dict['wav_path'] = result_dict['path']
            result_dict['wav'] = audio2tensor(result_dict['path'])
        else:
            print('error path:', result_dict['path'])
            assert(False)
            
        if 'AUD_' in result_dict['name']:
            return result_dict
        #load and crop video
        imgs_list = video2sequence(result_dict['path'], return_path=False, o_code='fan_in')
        #imgs = torch.stack(imgs_list, dim=0)
        with torch.no_grad():
            #result_dict['imgs'] = convert_img(self.fan.crop(imgs_list), 'fan_out', 'store') # OK
            # save emo tensor and params only
            crop_imgs = self.fan.crop(imgs_list)
            if crop_imgs is None:
                print('error unable to crop ', result_dict['name'])
            else:
                imgs = convert_img(crop_imgs, 'fan_out', 'store')
                result_dict['imgs'] = imgs
                result_dict['emo_tensor'] = self.dan.inference(convert_img(imgs, 'store', 'dan'))
                #generate codedict(for output norm)
                #print('before encode:', torch.cuda.max_memory_allocated(self.emoca.device) / (1024*1024*1024), ' GB')
                result_dict['code_dict'] = \
                    self.emoca.encode(convert_img(imgs, 'store','emoca').to(self.emoca.device), return_img=False) # dict(seq_len, code)
                for key in result_dict['code_dict'].keys():
                    result_dict['code_dict'][key] = result_dict['code_dict'][key].detach().to('cpu')
                result_dict['params'] = torch.cat([result_dict['code_dict']['expcode'], result_dict['code_dict']['posecode']], dim=-1)
        # calculate frame rate
        if 'params' in result_dict.keys():
            fps = round(result_dict['params'].size(0) / (result_dict['wav'].size(0) / 16000))
            print('fps estimated is ', fps)
            result_dict = adjust_frame_rate(result_dict, fps)
        else:
            print('no fps, default is 30')
        return result_dict
    
    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    data_path = '/home/chenyutong/facialanimation/dataset_cache/VOCASET'
    label_dict = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1}
    dataset = VOCASET(data_path=data_path, dataset_type='train', device=None, debug=0)
    dataset = VOCASET(data_path=data_path, dataset_type='test', device=None, debug=0)
    data = dataset[0]
    '''
    shape: [1,100] tex: [1,50] exp: [1,50] pose: [1,6] cam: [1,3] light:[1,9,3] images:[1,3,224,224] detail:[1,128]
    params: 50+6=56
    '''
    print('label size, wav size, params size:')
    print(data['wav'].size())
    print(data['params'].size())
