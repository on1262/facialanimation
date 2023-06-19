import torch
import sys
import os.path as path
import os
from math import ceil
import numpy as np
import random
from torch.utils.data import RandomSampler, SequentialSampler
from utils.flexible_loader import FlexibleLoader
import torch.nn as nn
from datetime import datetime, date
from utils.loss_func import ParamLossFunc, EmoTensorPredFunc, MouthConsistencyFunc
from utils.grad_check import GradCheck
from utils.converter import save_img, convert_img
from fitting.fit_utils import Mesh
from fitting.fit import approx_transform_mouth, get_mouth_landmark
import importlib
from threading import Thread, Event
from queue import Queue
from dataset import FACollate_fn, EnsembleDataset, zero_padding, BaselineVOCADataset
from utils.interface import EMOCAModel, DANModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.scheduler import PlateauDecreaseScheduler
from utils.mem_check import MemCheck
from utils.balance_data import cal_hist
from plyfile import PlyData
from utils.interface import FLAMEModel
#from memory_profiler import profile


def vertices2nparray(vertex):
    return np.asarray([vertex['x'],vertex['y'],vertex['z']]).T

class Trainer():
    def __init__(self, 
    imbalance_sample=True, 
    load_path=None, 
    save_path='/home/chenyutong/facialanimation/Model', 
    wav2vec_path='/home/chenyutong/facialanimation/wav2vec2/pretrained_model',
    model_name='cnn_lstm',
    dataset_path=None,
    label_dict={'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1},
    version=0,
    dev_str='cuda:0',
    batch_size=128,
    log_samples=True,
    debug=0
    ):
        self.debug = debug
        self.log_samples = log_samples
        self.model_name = model_name
        print('model name: ', model_name)
        print('debug mode: ', 'no debug' if debug == 0 else str(debug))
        # import model
        self.model_path = '/home/chenyutong/facialanimation/Model/'
        Model = importlib.import_module('Model.' + model_name + '.model').Model
        Config = importlib.import_module('Model.' + model_name + '.config').Config
        
        if self.debug > 0:
            torch.autograd.set_detect_anomaly(True)
        
        # save configs
        print('Trainer: loading model')
        self.device=torch.device(dev_str[0])
        self.mc1 = MemCheck(True, self.device)
        if len(dev_str) > 1:
            self.other_devices = [torch.device(dev_i) for dev_i in dev_str[1:]]
            self.mc2 = MemCheck(True, self.other_devices[0])
        else:
            self.mc2 = MemCheck(True, self.device)
            self.other_devices = None
        self.batch_size = batch_size
        self.model = None
        if load_path is not None:
            self.load(load_path)
        else:
            config = Config()
            config.args['wav2vec_path'] = wav2vec_path
            config.args['debug'] = debug
            self.model = Model(*config.parse_args(version=version))
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('model total_params:', total_params)
        self.model = self.model.to(self.device)


        test_voca_path = r'/home/chenyutong/facialanimation/dataset_cache/VOCASET'
        self.test_voca_dataset = BaselineVOCADataset(test_voca_path, device=self.device)
        self.flame = FLAMEModel(self.device)

        # init scheduler
        if 'emo' in self.model_name:
            self.lstm_epoch = 10 # actually it is lstm_epoch + 1 = 2, this is enough for lstm
        self.lr_config = self.get_lr_config(self.model_name)
        self.sche_type = self.lr_config['sche_type']
        if self.sche_type == 'ReduceLROnPlateau':
            self.sche_list = []
            for opt in self.model.get_opt_list():
                for g in opt.param_groups:
                    g['lr'] = self.lr_config['init']
                self.sche_list.append(ReduceLROnPlateau(opt, \
                    factor=self.lr_config['factor'], min_lr=self.lr_config['min'], patience=3, verbose=True, threshold=1e-5))
        elif self.sche_type == 'PlateauDecreaseScheduler':
            self.sche_list = [
                PlateauDecreaseScheduler(self.model.get_opt_list(),
                    lr_coeff_list=self.lr_config['lr_coeff_list'],
                    warmup_steps=self.lr_config['warmup_steps'],
                    warmup_lr=self.lr_config['warmup_lr'],
                    warmup_enable_list= self.lr_config['warmup_enable_list'],
                    factor=self.lr_config['factor'],
                    init_lr=self.lr_config['init_lr'],
                    min_lr=self.lr_config['min_lr'],
                    patience=self.lr_config['patience']
                    )
            ]
        # init dataset
        print('Trainer: init dataset')
        self.label_dict = label_dict
        self.dataset_path = dataset_path
        # init other model
        if 'emo' in self.model_name:
            self.dan = DANModel(device=self.device)
            self.emoca = EMOCAModel(device=self.other_devices[0] if self.other_devices is not None else self.device, decoder_only=False)
            self.model.set_emoca(self.emoca) # register emoca
        else:
            self.emoca = None
        dataset_dev = self.emoca.device if self.emoca is not None else self.device
        self.dataset_train = EnsembleDataset(
            self.dataset_path, self.label_dict, return_domain=True, dataset_type='train',device=dataset_dev, emoca=self.emoca, debug=debug)
        self.dataset_test = EnsembleDataset(
            self.dataset_path, self.label_dict, return_domain=False, dataset_type='test',device=dataset_dev, emoca=self.emoca, debug=debug)

        self.emotion_class = 6
        tr_sampler = RandomSampler(self.dataset_train)
        self.train_loader = FlexibleLoader(self.dataset_train, batch_size=batch_size, sampler=tr_sampler, collate_fn=FACollate_fn)
        # if batch size for test dataset > 1, best / worst samples can not be located exactly. 
        te_sampler = SequentialSampler(self.dataset_test)
        self.test_loader = FlexibleLoader(self.dataset_test, batch_size=16, sampler=te_sampler, collate_fn=FACollate_fn)

        print('output norm')
        self.norm_dict = self.calculate_norm(self.dataset_train)
        self.model.set_norm(self.norm_dict['model_norm'], self.device)
        if 'emo' in self.model_name:
            self.dan.set_norm(self.norm_dict['dan_norm'])
        print('Trainer: load dataset: done')
    
        # init criterion
        if 'emo' in self.model_name:
            self.cri_vert = ParamLossFunc()
            self.cri_vert.set_hist(self.norm_dict['param_hist_list'])
            self.cri_pred = EmoTensorPredFunc()
            self.cri_pred.set_hist(self.norm_dict['dan_hist_list'])
            self.cri_mouth = MouthConsistencyFunc()
            self.cri_mouth.set_hist(self.norm_dict['param_hist_list'])
        else:
            self.cri_vert = ParamLossFunc()
            self.cri_attns = nn.L1Loss(reduction='sum')

        # init others
        self.queue_stream = Queue()
        self.queue = Queue()

        self.registered_loss = {'out_loss' : 0} # out loss may not be loss for backward

        self.plot_folder = '/home/chenyutong/facialanimation/figures'
        self.gradcheck = GradCheck(self.model, (self.model_name + '_' + str(self.batch_size)) if self.debug == 0 else self.model_name + '-debug', plot=True, plot_folder=self.plot_folder)
        self.fps = 30 # align with audio
        self.save_path = path.join(save_path, 'saved_model')
        self.epoch = 0
        os.makedirs(self.save_path, exist_ok=True)

    def calculate_norm(self, dataset):
        output = {}
        if os.path.exists('/home/chenyutong/facialanimation/dataset_cache/new_norm.pt'):
            print('load from new_norm.pt')
            sd_dataset = torch.load('/home/chenyutong/facialanimation/dataset_cache/new_norm.pt')
            param_norm = sd_dataset['norm']
            param_hist_list = sd_dataset['hist_list']
        else:
            params = []
            for idx in range(len(dataset)):
                data = dataset[idx]
                params.append(data['params'].to('cpu'))
                #data.clear()
            params = torch.cat(params, dim=0)
            param_norm = {'max' : torch.max(data['params'], dim=0).values, 'min' : torch.min(data['params'], dim=0).values}
            param_hist_list = cal_hist(params.permute(1,0),bins=15, save_fig=True, name_list=['param_' + str(idx) for idx in range(56)])
            torch.save({'norm':param_norm, 'hist_list':param_hist_list}, '/home/chenyutong/facialanimation/dataset_cache/new_norm.pt')
        output['model_norm'] = param_norm
        output['param_hist_list'] = param_hist_list
        #param_norm['min'][53] = 0 # jaw angle
        print('max: ', param_norm['max'].max(dim=0).values, 'min: ', param_norm['min'].min(dim=0).values)
        if 'emo' in self.model_name:
            if os.path.exists('/home/chenyutong/facialanimation/DAN/dan_norm.pt'):
                print('load from dan_norm.pt')
                sd = torch.load('/home/chenyutong/facialanimation/DAN/dan_norm.pt')
                dan_out_norm = sd['norm']
                dan_hist_list = sd['hist_list']
            else:
                dan_norm = []
                for idx in range(len(dataset)):
                    data = dataset[idx]
                    if 'emo_tensor' not in data.keys():
                        data['emo_tensor'] = self.dan.inference(convert_img(data['imgs'], 'store', 'dan')).detach() # batch, max_seq_len, 7
                    dan_norm.append(data['emo_tensor'].to('cpu'))
                dan_norm = torch.cat(dan_norm, dim=0)
                dan_mean = torch.mean(dan_norm, dim=0)
                dan_std = torch.std(dan_norm, dim=0)
                dan_out_norm = torch.stack([dan_mean, dan_std])
                # cal hist after norm
                dan_hist_list = cal_hist(dan_norm.permute(1,0), bins=15, save_fig=True, name_list=self.dan.labels)
                torch.save({'norm':dan_out_norm,'hist_list':dan_hist_list}, '/home/chenyutong/facialanimation/DAN/dan_norm.pt')
            print('dan norm:', dan_out_norm)
            output['dan_norm'] = dan_out_norm
            output['dan_hist_list'] = dan_hist_list
        return output

    def save(self):
        epoch = self.epoch
        # generate name by date and time
        today = date.today()
        da = today.strftime("%b-%d")
        now = datetime.now()
        current_time = now.strftime("%H-%M")
        filename = 'date-' + da + '-time-' + current_time + '-epoch-' + str(epoch) + '.pth'
        torch.save({
            'model' : self.model.state_dict(),
            'epoch' : self.epoch,
            'model-config' : self.model.get_configs(),
            'norm' : self.norm_dict
            }, path.join(self.save_path, filename))
        print('model saved as ', filename)
        return path.join(self.save_path, filename)

    def load(self, model_path):
        print('load model from: ', model_path)
        save_dir = os.path.dirname(os.path.dirname(model_path))
        model_name = os.path.split(save_dir)[1]
        Model = importlib.import_module('Model.' + model_name + '.model').Model
        #Config = importlib.import_module('Model.' + model_name + '.config').Config
        sd = torch.load(model_path)
        self.epoch = sd['epoch']
        self.model = Model.from_configs(sd['model-config'])
        self.model.load_state_dict(sd['model'])
        del sd # release memory
        self.model.to(device=self.device)
        self.model.set_emoca(self.emoca)

    def get_loss_item(self, outs, gt):
        out_dict = {}

        if 'emo' in self.model_name:
            #print('out verts:', outs['verts'].size())
            vert_mask = outs['mask'][:, :, [0]].unsqueeze(-1).expand(-1,-1, 5023,3)
            vert_loss = self.cri_vert.cal_loss(outs['verts'], gt['verts'], out_mask=vert_mask) * 1000 # output after fixer
            # vert_loss_ori = self.cri_params.cal_loss(outs['params_ori'], gt['params'], out_mask=outs['mask']) # output before fixer
            #param_loss = self.cri_params.cal_loss(outs['params'], gt['params'], out_mask=None)
            out_dict['vert'] = vert_loss.detach().cpu().item()
            m_jaw_loss,m_jaw_vol_loss, m_param_loss, m_vertex_loss = \
                self.cri_mouth.cal_loss(outs['params'], gt['params'], outs['verts'], gt['verts'], gt['wav'], out_mask=outs['mask'])
            out_dict['m_jaw'] = m_jaw_loss.detach().cpu().item()
            out_dict['m_jaw_vol'] = m_jaw_vol_loss.detach().cpu().item()
            out_dict['m_param'] = m_param_loss.detach().cpu().item()
            out_dict['m_vertex'] = m_vertex_loss.detach().cpu().item()
            avg_loss = vert_loss + m_jaw_loss + m_jaw_vol_loss + m_vertex_loss + m_param_loss
            if 'emo_tensor' in gt.keys():
                # pred_loss = self.cri_pred.cal_loss(outs['pred_emo_tensor'], gt['emo_tensor'])
                pred_loss = self.cri_pred.cal_loss(outs['pred_emo_tensor'], gt['emo_tensor'], gt['seqs_len'])
                out_dict['pred'] = pred_loss.detach().cpu().item()
                avg_loss += pred_loss
            out_dict['out_loss'] = avg_loss.detach().cpu().item()

        return avg_loss, out_dict

    def preload(self, device, convert_cuda_list, queue:Queue):
        it = queue.get()
        try:
            data = next(it)
        except StopIteration as e:
            queue.put(None)
            return

        for key in data.keys():
            if key in convert_cuda_list:
                if isinstance(data[key], list):
                    data[key] = [item.to(device) for item in data[key]]
                elif isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
        # preprocess data
        if 'emo' in self.model_name:
            if 'imgs' in data.keys():
                data['imgs'] = torch.cat(data['imgs'], dim=0)
                if 'emo_tensor' not in data.keys():
                    data['emo_tensor'] = self.dan.inference(convert_img(data['imgs'], 'store', 'dan')).detach().to(self.device) # batch, max_seq_len, 7
            data['verts'] = self.emoca.decode(data['code_dict'], {'verts'}, target_device=self.device)['verts']
            
            self.mc1.log('after-gt-emo-tensor')
            if self.debug == 2: # single step
                save_img(convert_img(data['imgs'][0][0,...], i_code='store', o_code='tvsave'), 'ori.png')
            #data.pop('imgs')
        if 'gst' in self.model_name:
            # prepare gst data
            with torch.no_grad():
                wav = data['wav'].to(self.device)
                wav_len = torch.round(data['seqs_len'] * (16000/30))
                wav_len = torch.clip(wav_len, None, data['wav'].size(0))
                #print('wav', wav.size(), 'wav_len', wav_len.size())
                data['style_ebd'] = self.gst.inference(wav, wav_len, in_sr=16000)
        queue.put(data)

        return

    def run_epoch(self):
        # set seed
        SEED = 100
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        print('=======epoch: ', self.epoch, '=======')
        if self.lr_config['sche_type'] == 'PlateauDecreaseScheduler':
            print('current lr=', self.sche_list[0].get_lr(), 'wmp step=', self.sche_list[0].get_wmp_step())
        else:
            print('current lr=', self.model.get_opt_list()[0].param_groups[0]['lr'])
        train_avg_loss = 0
        test_max_loss = 0
        test_avg_loss = 0
        lmk_idx_out = get_mouth_landmark('flame')
        lmk_idx_gt = get_mouth_landmark('flame')
        self.model = self.model.eval()
        templates = {}

        with torch.no_grad():
            for idx,train_data in enumerate(self.test_voca_dataset):
                d = {'wav':train_data['wav'].to(self.device), 'code_dict':None , 'name':train_data['name'], 'seqs_len':train_data['seqs_len'],
                    'flame_template':train_data['flame_template'], 'shapecode':train_data['shapecode'].to(self.device)}
                d['emo_tensor_conf'] = ['no_use']
                gt = train_data['verts']
                seq_len = gt.size(0)
                params =  self.model.test_forward(d)['params'].squeeze(0) # 1, vertexm 3
                codedict = {'shapecode':torch.zeros((params.size(0), 100), device=self.device), 'expcode':params[:,:50], 'posecode':params[:,50:]}
                codedict['shapecode'] = train_data['shapecode'].to(self.device)
                output = self.flame.forward(codedict)
                output = torch.nn.functional.interpolate(output.unsqueeze(0).permute(0,2,1,3), size=(gt.size(0),3)).permute(0,2,1,3).squeeze(0)
                
                gt, output = gt.detach().cpu().numpy(), output.detach().cpu().numpy()
                output = np.asarray(output)
                if d['flame_template']['ply'] not in templates.keys():
                    tmp = PlyData.read(d['flame_template']['ply'])
                    templates[d['flame_template']['ply']] = Mesh(vertices2nparray(tmp['vertex']),'flame')
                output = output + (templates[d['flame_template']['ply']].v - output[0,:,:])

                seq_max_loss = 0
                seq_avg_loss = 0
                for idx2 in range(seq_len):
                    m_out = Mesh(output[idx2,:,:], 'flame')
                    m_gt = Mesh(gt[idx2,:,:], 'flame')
                    m_out = approx_transform_mouth(m_out, m_gt)
                    #m_out = mesh_seq[idx2]
                    delta = m_gt.v[lmk_idx_gt,:]-m_out.v[lmk_idx_out,:]
                    delta = np.sqrt(np.power(delta[:,0],2) + np.power(delta[:,1],2) + np.power(delta[:,2],2))
                    seq_avg_loss += np.mean(delta)
                    seq_max_loss += np.max(delta)
                test_max_loss += (seq_max_loss / seq_len)
                test_avg_loss += (seq_avg_loss / seq_len)
                if idx % 10 == 0:
                    print('idx=',idx, 'mean loss=', test_avg_loss/(idx+1), 'max loss=', test_max_loss/(idx+1), 'name=', train_data['name'])
        test_max_loss = test_max_loss / len(self.test_voca_dataset)
        test_avg_loss = test_avg_loss / len(self.test_voca_dataset)
        print(f'max loss={test_max_loss}, avg_loss={test_avg_loss}')
        
        self.model = self.model.train()
        for key in self.registered_loss.keys():
            self.registered_loss[key] = 0
        
        # init dataloader
        dataloader = self.train_loader
        if self.debug > 0:
            dataloader.batch_size = 16
        else:
            dataloader.batch_size = 100
        idx_it = 0
        it = iter(dataloader)
        convert_cuda_list = {'emo_label', 'wav', 'params', 'emo_tensor'} # do not convert imgs into gpu
        
        self.queue.put(it)
        t_preload = Thread(target=self.preload, args=(self.device, convert_cuda_list, self.queue))
        t_preload.start()
        t_preload.join()
        train_data = self.queue.get()

        while train_data is not None:
            self.mc1.log('iter-start')
            for opt in self.model.get_opt_list():
                opt.zero_grad(set_to_none=True)
            self.queue.put(it)
            t_preload = Thread(target=self.preload, args=(self.device, convert_cuda_list, self.queue))
            t_preload.start()
            if self.debug > 0: # encoder needed, avoid out of memory when EMOCA encoder is running at the same time of dan or model
                t_preload.join()
            # load background images
            #for idx in range(len(data['code_dict'])):
            #    data['code_dict'][idx].update({'images': convert_img(data['imgs'][idx], 'store', 'emoca').to('cpu')})
            # generate emo tensor
            self.mc1.log('before-forward')
            self.mc2.log('before-forward-emoca')
            
            # if 'emo' in self.model_name:
                #data_verts = []
                #for codedict in data['code_dict']:
                #    data_verts.append(self.emoca.decode(codedict, {'verts'}, target_device=self.device)['verts'])
                #data['verts'] = torch.cat(data_verts, dim=0)
                
            
            out_dict = self.model.batch_forward(train_data) # forward
            
            self.mc1.log('after-forward')
            self.mc2.log('after-forward-emoca')
            if 'emo' in self.model_name:
                out_dict['verts'] = self.emoca.decode(out_dict['code_dict'], {'verts'}, target_device=self.device)['verts']
                #out_verts = []
                #for codedict in out_dict['code_dict']:
                #    out_verts.append(self.emoca.decode(codedict, {'verts'}, target_device=self.device)['verts'])
                #out_dict['verts'] = torch.cat(out_verts, dim=0)
                

                
            
            self.mc1.log('before-cal-loss')
            
            loss, loss_dict = self.get_loss_item(out_dict, train_data)
            self.mc1.log('after-cal-loss')

            for key in loss_dict.keys():
                if self.registered_loss.get(key) is None: # first iteration only
                    self.registered_loss[key] = 0
                self.registered_loss[key] += loss_dict[key]

            
            # backward
            self.mc1.log('before-backward')
            loss.backward()
            self.mc1.log('after-backward')
            if self.epoch % 2 == 0 and idx_it==0: # execute every 5 epoch
                self.gradcheck.check_grad(disp=True)
            #optimize
            if idx_it % ceil(128/dataloader.batch_size) == 0:
                for opt in self.model.get_opt_list():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    self.sche_list[0].step(self.epoch)
            self.mc1.log('after-opt-step')
            del loss
            
            if self.debug == 0: # async
                t_preload.join()
            train_data = self.queue.get()
            idx_it += 1
            self.mc1.log('iter-end')
            if len(dataloader) >= 5 and idx_it % round(len(dataloader)/5) == 0 and idx_it != 0: # execute 5 times per epoch
                #self.mc1.summary()
                #self.mc2.summary()
                print('iter ', idx_it, [key + '=' + str(self.registered_loss[key] / (idx_it+1)) for key in self.registered_loss.keys()])
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            self.mc1.clear()
            self.mc2.clear()
        print('registered loss:', [key + '=' + "{:.3f}".format(self.registered_loss[key] / len(dataloader)) for key in self.registered_loss.keys()])
        train_avg_loss = self.registered_loss['out_loss'] / len(dataloader)

        # if self.sche_type == 'ReduceLROnPlateau':
        #     for sche in self.sche_list:
        #         sche.step(test_max_loss) # TODO
        # elif self.sche_type == 'PlateauDecreaseScheduler':
        #     self.sche_list[0].step(train_avg_loss, test_max_loss, self.epoch)
        self.epoch += 1
        return test_max_loss
        

    def get_lr_config(self, model_name):
        lr_config = {}
        lr_config['sche_type'] = 'PlateauDecreaseScheduler'
        lr_config['lr_coeff_list'] = [1, 1]
        lr_config['warmup_steps'] = 200
        lr_config['warmup_lr'] = 1e-3
        lr_config['warmup_enable_list'] = [True, True]
        lr_config['factor'] = 0.2
        lr_config['init_lr'] = 1e-4
        lr_config['min_lr'] = 1e-4
        lr_config['patience'] = 3
        return lr_config
        
        
