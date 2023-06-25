#!/home/chenyutong/facialanimation
import argparse
import glob
import importlib
import os
import random
import subprocess
#from utils.loss_func import LossFunc
import sys

import numpy as np
import torch
from tqdm import tqdm

from dataset import (CREMADDataset, FACollate_fn, TESTDataset, get_emo_label_from_name)
from utils.converter import (audio2tensor, convert_img, save_img, video2sequence, video2wav)
from utils.config_loader import GBL_CONF, PATH
from utils.emo_curve_check import plot_curve
from utils.fitting.fit_utils import Mesh
from utils.flexible_loader import FlexibleLoader
from utils.interface import DANModel, EMOCAModel, FaceFormerModel
from utils.generic import multi_imgs_2_video, load_model_dict

def get_gt_from_dataset(dataset_path, load_name):
    # get params from .pt dataset file
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
    def __init__(self):
        self.infer_conf = GBL_CONF['inference']
        self.in_dir = PATH['inference']['input']
        self.out_dir = PATH['inference']['output']
        self.cache_dir = PATH['inference']['cache']
        self.device = torch.device(GBL_CONF['global']['device'])
        self.mode = self.infer_conf['mode']
        self.emo_cls = self.infer_conf['infer_dataset']['emo_cls']
        torch.cuda.set_device(self.device) # removing this line cause CUDA illegal memory assess, see https://github.com/pytorch/pytorch/issues/21819
        self.emoca = EMOCAModel(self.device)
        self.ffm = FaceFormerModel(self.device)
        model_path = os.path.join(PATH['model'], self.infer_conf['mode_conf']['model_name'])
        self.model = self.load_model(model_path, self.emoca, self.device)


    def load_model(self, model_path, emoca, device):
        sd, Model = load_model_dict(model_path, device)
        self.model = Model.from_configs(sd['model-config'])
        self.model.load_state_dict(sd['model'])
        self.model.to(device=device)
        self.model.set_emoca(emoca)
        return self.model

    def get_output_mask(self, loss_mask):
        return torch.ones(loss_mask.size(), device=loss_mask.device) * (loss_mask > 0)

    def forward(self, data):
        out_dict = self.model.test_forward(data) # forward
        return out_dict

    def run(self):
        # label_dict = {'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5, 'EXC':1}
        '''
        config format:
        video: use video to reconstruct 3DMM, add speech driven result
        emo-cls: input audio, output varying emotion class
        emo-ist: varying intensity
        audio: input audio, output animation
        '''
        sample_configs = self.infer_conf['infer_dataset']['sample_configs']
        # vid_path: dir, aud_path: dir
        print('loading dataset...')
        # debug = 1, disable auto parse
        dataset = TESTDataset(device=self.device, emoca=self.emoca)
        for data in dataset:
            conf = sample_configs[data['name']]
            print('processing ', data['name'], 'conf=', conf)

            # create and clear dirs
            sample_cache_dir = os.path.join(self.cache_dir, data['name'] + conf)
            sample_out_dir = os.path.join(self.out_dir, data['name'] + conf)
            subprocess.run(['rm', '-rf', sample_cache_dir])
            subprocess.run(['rm', '-rf', sample_out_dir])
            os.makedirs(sample_cache_dir, exist_ok=True)
            os.makedirs(sample_out_dir, exist_ok=True)
        
            with torch.no_grad():
                data['wav'] = data['wav'].to(self.device)
                data['smooth'] = self.infer_conf['smoothing']
                # data codedict->imgs
                if 'video' in conf:
                    # generate reconstructed images from EMOCA code dict
                    ori_imgs = convert_img(data['imgs'], 'store', 'tvsave').to('cpu')
                    data['emoca_imgs'] = self.emoca.decode(data['code_dict'], {'coarse'}, target_device=self.device)['output_images_coarse']
                    data['emoca_imgs'] = convert_img(data['emoca_imgs'], 'emoca', 'tvsave').to('cpu')
                    # generate ground truth emotion logits
                    emo_logits_gt = self.dan.inference(convert_img(data['imgs'], 'store', 'dan')).detach().to(self.device) # batch, max_seq_len, 7
                    plot_curve(emo_logits_gt, self.dan.labels, os.path.join(sample_out_dir, data['name'] + '_ori.png'))
                # clear ground truth
                if 'code_dict' in data.keys():
                    print('reset code dict')
                    data['code_dict']['posecode'].zero_()
                    data['code_dict']['shapecode'].zero_()
                    data['code_dict']['expcode'].zero_()
                
                out_et_emo = {}
                out_speech_driven = None
                if 'video' in conf:
                    state_list = ['_et_ori', '_et_speech_driven', 'no_emo']
                elif 'audio' in conf:
                    state_list =  ['_et_speech_driven','no_emo']
                elif 'aud-cls' in conf:
                    state_list = [conf.split('=')[-1] ,'no_emo', 'faceformer'] # e.g. conf.split('=')[-1] = 'HAP'
                elif 'emo-cls' in conf:
                    state_list = self.emo_cls
                elif 'emo-ist' in conf:
                    state_list = ['HAP-1.0', 'HAP-0.5', 'HAP-0.25', 'FEA-0.25', 'FEA-0.5','FEA-1.0']
                
                for state in state_list:
                    if state == 'faceformer':
                        code_dict = {key:p[[0],:] for key, p in data['code_dict'][0].items()}
                        emoca_out = self.emoca.decode(code_dict, {'verts'}, target_device=self.device) # generate template with all code zero
                        verts = emoca_out['verts'].detach()
                        self.ffm.template[data['name']] = torch.reshape(verts, (1, 5023*3))
                        with torch.no_grad():
                            fv = self.ffm.forward(data)
                            print('faceformer output: ', fv.size(), 'seqs_len:', data['seqs_len'][0])
                            if fv.size(0) > data['seqs_len']:
                                fv = fv[:data['seqs_len'],...]
                            elif fv.size(0) < data['seqs_len']:
                                delta_fv = fv[[-1],...].repeat((data['seqs_len'] - fv.size(0), 1, 1))
                                fv = torch.cat([fv, delta_fv])
                            data['code_dict'][0]['decode_verts'] = fv
                            out_dict['imgs'] = self.emoca.decode(data['code_dict'], {'geo', 'decode_verts'}, target_device=self.device)['geometry_coarse']
                    elif state == '_et_speech_driven': # emotion information will be predicted from audio
                        data['emo_logits_conf'] = ['use']
                        data['emo_logits'] = None
                        data['emo_label'] = None
                    elif state == 'no_emo': # do not predict emotion from audio
                        data['emo_logits_conf'] = ['no_use']
                        data['emo_logits'] = None
                        data['emo_label'] = None
                    elif state in self.emo_cls: # add custom emotion
                        data['emo_logits_conf'] = ['one_hot']
                        data['emo_logits'] = None
                        data['emo_label'] = state
                        data['intensity'] = self.infer_conf['default_intensity']
                    elif state.split('-')[0] in self.emo_cls: # specify emoiton class and intensity
                        data['emo_logits_conf'] = ['one_hot']
                        data['emo_logits'] = None
                        data['emo_label'] = state.split('-')[0]
                        data['intensity'] = float(state.split('-')[-1])
                    elif state == '_et_ori': # use emotion logits from original video
                        data['emo_logits_conf'] = ['use']
                        assert(data.get('emo_logits') is not None)
                        data['emo_label'] = None

                    if state != 'faceformer':
                        out_dict = self.model.test_forward(data) # forward
                        if state == '_et_speech_driven': # plot predicted emotion logits
                            plot_curve(data['emo_logits'], self.dan.labels, os.path.join(self.sample_out_dir, data['name'] + '_pred.png'))
                        if 'tex' in conf: # use texture extracted from input video
                            out_dict['imgs'] = self.emoca.decode(
                                out_dict['code_dict'], {'coarse'}, target_device=self.device)['output_images_coarse']
                        else:
                            emoca_out = self.emoca.decode(out_dict['code_dict'], {'geo'}, target_device=self.device)
                            out_dict['imgs'] = emoca_out['geometry_coarse']

                    out_dict['imgs'] = convert_img(out_dict['imgs'], 'emoca', 'tvsave').to('cpu')
                    if state == '_et_ori':
                        out_et_ori = out_dict['imgs']
                    elif state == 'faceformer':
                        out_ff = out_dict['imgs']
                    elif state == 'no_emo':
                        out_no_emo = out_dict['imgs']
                    elif state in self.emo_cls:
                        out_et_emo[state] = out_dict['imgs']
                    elif state == '_et_speech_driven':
                        out_speech_driven = out_dict['imgs']
                    elif state.split('-')[0] in self.emo_cls:
                        out_et_emo[state] = out_dict['imgs']
                # save result
                for i in range(data['code_dict'][0]['shapecode'].size(0)): # iterate frames
                    if 'video' in conf:
                        frame_img = [ori_imgs[i,...], data['emoca_imgs'][i,...], out_et_ori[i,...], out_speech_driven[i,...], out_no_emo[i,...]]
                    elif 'audio' in conf:
                        frame_img = [out_speech_driven[i,...], out_no_emo[i,...]]
                    elif 'emo-cls' in conf:
                        frame_img = [out_et_emo[key][i,...] for key in state_list]
                    elif 'aud-cls' in conf:
                        frame_img = [out_et_emo[state_list[0]][i,...], out_no_emo[i,...], out_ff[i,...]]
                    elif 'emo-ist' in conf:
                        frame_img = [out_et_emo[key][i,...] for key in state_list]
                    save_img(torch.cat(frame_img, -1), os.path.join(sample_cache_dir, '%04d.jpg' % i))
                multi_imgs_2_video(os.path.join(sample_cache_dir, '%04d.jpg'), data['wav_path'], os.path.join(sample_out_dir, data['name'] + '.mp4'))
        print('Inference Done')
