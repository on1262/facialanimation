#!/home/chenyutong/facialanimation
import torch
import argparse
import os
import subprocess
#from utils.loss_func import LossFunc
import sys
import importlib
import glob
import random
import numpy as np
from utils.flexible_loader import FlexibleLoader
from tqdm import tqdm
from utils.interface import EMOCAModel, DANModel
from dataset import TESTDataset, FACollate_fn, CREMADDataset
from utils.emo_curve_check import plot_curve
from utils.converter import convert_img, save_img
from fitting.fit_utils import Mesh
from dataset import get_emo_label_from_name, get_emo_index
from utils.converter import video2sequence, audio2tensor, video2wav

def multi_imgs_2_video(img_path, audio_path, outputpath):
    if audio_path is not None and os.path.exists(audio_path):
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', \
            '-f', 'image2', '-r', '30', '-i', img_path, '-i', audio_path, '-b:v','1M', outputpath])
    else:
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2', '-r', '30', '-i', img_path, '-b:v','1M', outputpath])


# get params from .pt dataset file
def get_gt_from_dataset(dataset_path, load_name):
    data = torch.load(os.path.join(dataset_path, 'params.pt'), map_location='cpu')
    wavs = torch.load(os.path.join(dataset_path, 'wav.pt'), map_location='cpu')
    params_list = data['param']
    name_list = data['name']
    
    for idx,name in enumerate(name_list):
        if name == load_name:
            #torch.save(params_list[idx], save_path)
            print(load_name, ' found')
            return params_list[idx].unsqueeze(0), wavs[name]
    print('not found ' , load_name)

class Inference():
    def __init__(self, mode, device_str:str, model_path, Model):
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.mode = mode
        torch.cuda.set_device(self.device) # removing this line cause CUDA illegal memory assess, see https://github.com/pytorch/pytorch/issues/21819
        print('init emoca on ', self.device)
        sd = torch.load(model_path, map_location=self.device)
        self.emoca = EMOCAModel(device=self.device)
        self.dan = DANModel(device=self.device)
        self.model = Model.from_configs(sd['model-config'])
        self.model.load_state_dict(sd['model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        del sd
        print('init Done')


    def get_output_mask(self, loss_mask):
        return torch.ones(loss_mask.size(), device=loss_mask.device) * (loss_mask > 0)

    def forward(self, data):
        out_dict = self.model.test_forward(data) # forward
        return out_dict
    
    def create_matrix(self):
        '''生成CREMAD中若干序列的混淆矩阵, 需要对生成结果进行带纹理的渲染'''
        cre_path = '/home/chenyutong/facialanimation/dataset_cache/CREMA-D'
        label_dict={'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1}
        dan_label = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
        data_label = ['NEU', 'HAP', 'ANG', 'SAD', 'DIS', 'FEA']
        dataset = CREMADDataset(cre_path, label_dict, dataset_type='test', device=device, emoca=self.emoca, debug=0)
        acc_table = {}
        origin_table = {} # 测试集中各表情的数量
        with torch.no_grad():
            for img_idx, data in tqdm(enumerate(dataset), desc='Cremad dataset'):
                '''
                data: dict
                    'name': '1073_TIE_NEU_XX'
                    'emo_label': 0
                    'wav': tensor[36837]
                    'emo_tensor': [69. 7]
                    'code_dict': {shapecode, texcode, expcode, posecode, cam, lightcode}
                    'params': tensor[69, 56]
                '''
                data['wav'] = data['wav'].to(self.device)
                data['params'] = data['params'].to(self.device)
                data['emo_tensor'] = data['emo_tensor'].to(self.device)
                for key in data['code_dict']:
                    data['code_dict'][key] = data['code_dict'][key].to(self.device)

                origin_label = data_label[get_emo_label_from_name('cremad', label_dict, data['name'])]
                data['code_dict']['posecode'][:, 4:] = 0
                data['code_dict']['cam'][:,1] = 0
                data['code_dict']['cam'][:,2] = 0
                data['code_dict']['cam'][:,0] = 9 # stablize camera position
                data['code_dict']['posecode'][:, :3] = 0
                batch = FACollate_fn([data])
                batch['emo_tensor'] = None
                batch['emo_label'] = label_dict[origin_label]
                batch['emo_tensor_conf'] = ['one_hot']
                if origin_label == 'NEU':
                    batch['intensity'] = 1
                elif origin_label == 'SAD':
                    batch['intensity'] = 4
                else:
                    batch['intensity'] = 1
                batch['smooth'] = True
                # TODO adjust emotion
                out_dict = self.model.test_forward(batch)
                # generate images
                img = self.emoca.decode(
                    out_dict['code_dict'], {'coarse'}, target_device=self.device)['output_images_coarse']
                frames = random.choices(list(range(0, img.size(0))), k=10)
                img = img[frames, ...]
                # subprocess.run(['rm', '-rf', './matrix_out/*.jpg'])
                # clear cache
                # save_img(img, os.path.join('matrix_out', data['name'] + '.jpg'))
                # generate output label
                emo_out = self.dan.inference(convert_img(img, 'emoca', 'dan')).detach().to(self.device) # [80,7]
                # TODO 这里应该是投票法不是mean
                # coeff: ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
                coeff = torch.tensor([1, 1, 10, 0, 12, 8, 10], device=self.device)[None, :]
                emo_out = torch.exp(emo_out) * coeff
                out_idx = round(torch.median(torch.argmax(emo_out, dim=-1).to(float)).item())
                out_label = dan_label[out_idx]
                if origin_table.get(origin_label) is None:
                    origin_table[origin_label] = 1
                else:
                    origin_table[origin_label] += 1
                key = origin_label + '_' + out_label
                if key not in acc_table:
                    acc_table[key] = 1
                else:
                    acc_table[key] += 1
        # print matrix
        print('Emotion matrix: col=origin, row=predict')
        print('     ', data_label)
        for x in data_label: # origin type
            print(f'{x}:     ', end='')
            for y in data_label:
                key = x + '_' + y
                if acc_table.get(key) is None:
                    print('0/' + str(origin_table[x]), '\t\t', end='')
                else:
                    print(str(acc_table[key]) + '/' + str(origin_table[x]),'\t\t', end='')
            print('\n')
        print('Done')

    def create_trailer(self, vid_path, infer_sample_path, cache_path, model_name):
        # load faceformer
        from utils.interface import FaceFormerModel
        ffm = FaceFormerModel(self.device)
        # load video and model
        label_dict = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5, 'EXC':1}
        '''
        config format:
        video: use video to reconstruct 3DMM, add speech driven result
        emo-cls: input audio, output varying emotion class
        emo-ist: varying intensity
        audio: input audio, output animation
        '''
        sample_configs = {
            'hap_1' : 'video',
            'ang_1' : 'video',
            'test_1' : 'video',
            'flyme2themoon_ori': 'video',
            'flyme2themoon_2': 'emo-cls-tex',
            'flyme2themoon_3': 'emo-ist-tex',
            'AUD_faceformer_2': 'audio',
            'AUD_faceformer_3': 'audio',
            'AUD_earlier': 'emo-ist',
            'AUD_voca_1':'audio',
            'AUD_voca_2':'emo-cls',
            'AUD_F6_01_cut':'audio',
            'AUD_demo_2_happy' : 'aud-cls=HAP',
            'AUD_demo_2_angry' : 'aud-cls=ANG',
            'AUD_demo_2_sad' : 'aud-cls=SAD'
        }
        # vid_path: dir, aud_path: dir
        print('loading dataset...')
        # debug = 1, disable auto parse
        dataset = TESTDataset(vid_path, cache_path, label_dict, device=self.device, emoca=self.emoca)
        dataloader = FlexibleLoader(dataset, batch_size=1, sampler=None, collate_fn=FACollate_fn, clip_max=False)
        for data in dataloader:
            for key in data:
                if isinstance(data[key], list):
                    data[key] = data[key][0]
            print('processing ', data['name'])
            assert(sample_configs.get(data['name']) is not None)
            conf = sample_configs[data['name']]
            data_cache_path = os.path.join(cache_path, data['name'])
            subprocess.run(['rm', '-rf', data_cache_path])
            os.makedirs(data_cache_path, exist_ok=True)
        
            with torch.no_grad():
                data['wav'] = data['wav'].to(self.device)
                data['smooth'] = True # add smooth
                if 'video' in conf:
                    ori_imgs = convert_img(data['imgs'], 'store', 'tvsave').to('cpu')
                if 'code_dict' in data.keys():
                    data['code_dict']['posecode'][:, :3] = 0
                    data['code_dict']['posecode'][:, 4:] = 0
                    data['code_dict']['cam'][:,1] = 0
                    data['code_dict']['cam'][:,2] = 0
                    data['code_dict']['cam'][:,0] = 9 # stablize camera position
                # data codedict->imgs
                if 'video' in conf:
                    data['emoca_imgs'] = self.emoca.decode(
                        data['code_dict'], {'coarse'}, target_device=self.device)['output_images_coarse']
                    data['emoca_imgs'] = convert_img(data['emoca_imgs'], 'emoca', 'tvsave').to('cpu')
                    print('emoca imgs:', data['emoca_imgs'].size())
                    # no need to concat because data[key] = data[key][0]
                    emo_tensor_gt = self.dan.inference(convert_img(data['imgs'], 'store', 'dan')).detach().to(self.device) # batch, max_seq_len, 7
                    plot_curve(emo_tensor_gt, self.dan.labels, os.path.join(infer_sample_path, data['name'] + '_ori.png'))
                # modify code_dict in data
                if 'code_dict' in data.keys():
                    print('reset code dict')
                    data['code_dict']['posecode'].zero_()
                    data['code_dict']['shapecode'].zero_()
                    data['code_dict']['expcode'].zero_()
                out_et_emo = {}
                out_speech_driven = None
                if 'video' in conf:
                    state_list = ['_et_speech_driven', 'no_emo']
                elif 'audio' in conf:
                    state_list =  ['_et_speech_driven','no_emo']
                elif 'aud-cls' in conf:
                    state_list = [conf.split('=')[-1] ,'no_emo', 'faceformer']
                elif 'emo-cls' in conf:
                    state_list = ['NEU','ANG','HAP','SAD','DIS','FEA']
                elif 'emo-ist' in conf:
                    state_list = ['HAP-1.0', 'HAP-0.5', 'HAP-0.25', 'FEA-0.25', 'FEA-0.5','FEA-1.0']
                
                for emo_tensor_mode in state_list:
                    if emo_tensor_mode == 'faceformer':
                        code_dict = {key:p[[0],:] for key, p in data['code_dict'][0].items()}
                        emoca_out = self.emoca.decode(code_dict, {'verts'}, target_device=self.device)
                        verts = emoca_out['verts'].detach()
                        ffm.template[data['name']] = torch.reshape(verts, (1, 5023*3))
                        print('gt subject size:', verts.shape)
                    elif emo_tensor_mode == '_et_speech_driven': # audio input
                        #print('speech driven codedict:', data['code_dict'])
                        data['emo_tensor_conf'] = ['use']
                        data['emo_tensor'] = None
                        data['emo_label'] = None
                    elif emo_tensor_mode == 'no_emo':
                        data['emo_tensor_conf'] = ['no_use']
                        data['emo_tensor'] = None
                        data['emo_label'] = None
                    elif emo_tensor_mode in label_dict.keys(): # emo class
                        data['emo_tensor_conf'] = ['one_hot']
                        data['emo_tensor'] = None
                        data['emo_label'] = label_dict[emo_tensor_mode]
                        data['intensity'] = 0.5
                    elif emo_tensor_mode.split('-')[0] in label_dict.keys(): # emo class and intensity
                        data['emo_tensor_conf'] = ['one_hot']
                        data['emo_tensor'] = None
                        data['emo_label'] = label_dict[emo_tensor_mode.split('-')[0]]
                        data['intensity'] = float(emo_tensor_mode.split('-')[-1])
                    elif emo_tensor_mode == '_et_ori':
                        data['emo_tensor_conf'] = ['use']
                        assert(data.get('emo_tensor') is not None)
                        data['emo_label'] = None

                    if emo_tensor_mode == 'faceformer':
                        with torch.no_grad():
                            fv = ffm.forward(data)
                            print('faceformer output: ', fv.size(), 'seqs_len:', data['seqs_len'][0])
                            if fv.size(0) > data['seqs_len']:
                                fv = fv[:data['seqs_len'],...]
                            elif fv.size(0) < data['seqs_len']:
                                delta_0 = data['seqs_len'] - fv.size(0)
                                delta_fv = fv[[-1],...].repeat((delta_0, 1, 1))
                                fv = torch.cat([fv, delta_fv])

                            data['code_dict'][0]['decode_verts'] = fv
                            out_dict['imgs'] = self.emoca.decode(data['code_dict'], {'geo', 'decode_verts'}, target_device=self.device)['geometry_coarse']
                    else:
                        # process input emo Tensor
                        out_dict = self.model.test_forward(data) # forward
                        if emo_tensor_mode == '_et_speech_driven':
                            plot_curve(data['emo_tensor'], self.dan.labels, os.path.join(infer_sample_path, data['name'] + '_pred.png'))
                        # model out->imgs
                        if 'tex' in conf:
                            out_dict['imgs'] = self.emoca.decode(
                                out_dict['code_dict'], {'coarse'}, target_device=self.device)['output_images_coarse']
                            debug = False
                            if debug:
                                emoca_out = self.emoca.decode(out_dict['code_dict'], {'geo', 'verts'}, target_device=self.device)
                                out_dict['imgs'] = emoca_out['geometry_coarse']
                                verts = emoca_out['verts']
                                verts = np.asarray(verts.detach().cpu())
                                print('verts size:', verts.shape)
                                # print('verts max:', np.max(verts))
                                out_dir = os.path.join('test_output', conf + str(np.random.randint(1, 100)))
                                os.makedirs(out_dir, exist_ok=True)
                                for idx in range(verts.shape[0]):
                                    Mesh.write_obj('flame', verts[idx, :,:],  os.path.join(out_dir, str(idx) + '.obj'))
                        else:
                            emoca_out = self.emoca.decode(out_dict['code_dict'], {'geo'}, target_device=self.device)
                            out_dict['imgs'] = emoca_out['geometry_coarse']    
                        # print result
                    out_dict['imgs'] = convert_img(out_dict['imgs'], 'emoca', 'tvsave').to('cpu')
                    if emo_tensor_mode == '_et_ori':
                        out_et_ori = out_dict['imgs']
                    elif emo_tensor_mode == 'faceformer':
                        out_ff = out_dict['imgs']
                    elif emo_tensor_mode == 'no_emo':
                        out_no_emo = out_dict['imgs']
                    elif emo_tensor_mode in label_dict.keys():
                        out_et_emo[emo_tensor_mode] = out_dict['imgs']
                    elif emo_tensor_mode == '_et_speech_driven':
                        out_speech_driven = out_dict['imgs']
                    elif emo_tensor_mode.split('-')[0] in label_dict.keys():
                        out_et_emo[emo_tensor_mode] = out_dict['imgs']
                # save result
                plain_img = []
                for i in range(data['code_dict'][0]['shapecode'].size(0)):
                    if 'video' in conf:
                        frame_img = [ori_imgs[i,...], data['emoca_imgs'][i,...], out_speech_driven[i,...], out_no_emo[i,...] ]
                    elif 'audio' in conf:
                        frame_img = [out_speech_driven[i,...], out_no_emo[i,...]]
                    elif 'emo-cls' in conf:
                        frame_img = [out_et_emo[key][i,...] for key in state_list]
                    elif 'aud-cls' in conf:
                        frame_img = [out_et_emo[state_list[0]][i,...], out_no_emo[i,...], out_ff[i,...]]
                        if i % 10 == 0:
                            plain_img.append(frame_img)
                    elif 'emo-ist' in conf:
                        frame_img = [out_et_emo[key][i,...] for key in state_list]
                        if i % 10 == 0:
                            plain_img.append(frame_img)
                    save_img(torch.cat(frame_img, -1), os.path.join(data_cache_path, '%04d.jpg' % i))
                if 'emo-ist' in conf or 'aud-cls' in conf: # process plain image
                    save_img(torch.cat([torch.cat(f, -2) for f in plain_img],-1), os.path.join(infer_sample_path, data['name'] + '_plain.jpg'))
                multi_imgs_2_video(os.path.join(data_cache_path, '%04d.jpg'), data['wav_path'], os.path.join(infer_sample_path, data['name'] + '.mp4'))
        print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='2d', type=str, help='[3d, 2d, trailer, matrix]')
    parser.add_argument('--model', default='tf_emo_4', type=str, help='model name')
    parser.add_argument('--vid_path', default=None, type=str, help='video path')
    parser.add_argument('--cache_path', default=None, type=str, help='cache path')
    parser.add_argument('--infer_path', default=None, type=str, help='infer sample path')
    parser.add_argument('--gt_path', default=None, type=str, help='gt sample path')
    parser.add_argument('--device', default='cpu', type=str, help='n or cpu')
    args = parser.parse_args()
    
    work_dir = '/home/chenyutong/facialanimation'
    print('change work dir to' , work_dir)
    os.chdir(work_dir)
    sys.path.append(work_dir)
    Model = importlib.import_module('Model.' + args.model + '.model').Model
    
    
    print('clean output path and cache path')
    subprocess.run(['rm', '-rf', args.infer_path])
    if args.gt_path is not None:
        subprocess.run(['rm', '-rf', args.gt_path])
        os.makedirs(args.gt_path, exist_ok=True)
    
    os.makedirs(args.infer_path, exist_ok=True)
    if args.cache_path is not None:
        subprocess.run(['rm', '-rf', args.cache_path])
        os.makedirs(args.cache_path, exist_ok=True)
    subprocess.run(['rm', '-rf', os.path.join(args.vid_path,'frames')])

    print('init inference')
    device = 'cpu' if args.device == 'cpu' else 'cuda:' + args.device
    model_path = os.path.join('/home/chenyutong/facialanimation/Model', args.model,'saved_model')
    list_of_files = glob.glob(os.path.join(model_path, '*.pth')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print('load model from: ', latest_file)
    infer = Inference(args.mode, device, latest_file, Model)
    if args.mode == 'trailer':
        infer.create_trailer(args.vid_path, args.infer_path, args.cache_path, args.model)
    elif args.mode == 'matrix':
        print('create matrix')
        infer.create_matrix()
