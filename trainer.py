import importlib
import os
import os.path as path
from datetime import date, datetime
from math import ceil
from queue import Queue
from threading import Thread

import numpy as np
import torch
from plyfile import PlyData
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler, SequentialSampler

from dataset import BaselineVOCADataset, EnsembleDataset, FACollate_fn
from utils.balance_data import cal_hist
from utils.config_loader import GBL_CONF, PATH
from utils.converter import convert_img, save_img
from fitting import approx_transform_mouth, get_mouth_landmark, Mesh
from utils.flexible_loader import FlexibleLoader
from utils.generic import vertices2nparray, load_model_dict
from utils.grad_check import GradCheck
from utils.interface import DANModel, EMOCAModel, FLAMEModel
from utils.loss_func import EmoTensorPredFunc, MouthConsistencyFunc, ParamLossFunc
from utils.scheduler import PlateauDecreaseScheduler


class Trainer():
    def __init__(self):
        trainer_conf = GBL_CONF['trainer']
        self.model_path = os.path.join(PATH['model'], trainer_conf['model_name'])
        load_path = self.model_path if trainer_conf['load_from_checkpoint'] else None,

        self.debug = trainer_conf['debug']
        self.device=torch.device(GBL_CONF['global']['device'])
        self.preload_device = torch.device(GBL_CONF['global']['preload_device'])
        self.model_name = trainer_conf['model_name']
        self.fps = trainer_conf['fps']
        self.total_epoch = GBL_CONF['model'][self.model_name]['epochs']
        # load 3rd models
        self.flame = FLAMEModel(self.device)
        self.emoca = EMOCAModel(self.preload_device)
        self.dan = DANModel(device=self.device)
        
        print('model name: ', self.model_name)
        print('debug mode: ', self.debug == 0)
        print('Trainer: loading model')
        self.model = None
        if load_path is not None: # load existed model
            self.model, self.now_epoch = self.load_model(load_path, self.emoca, self.device)
        else: # create new model instance
            Model = importlib.import_module('models.' + self.model_name + '.model').Model
            self.model, self.now_epoch = Model(), 0
            self.model.set_emoca(self.emoca)
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('total params:', total_params)
        
        self.lr_config = GBL_CONF['model'][self.model_name]['lr_config']
        self.scheduler = self.lr_config['scheduler']
        if self.scheduler == 'ReduceLROnPlateau':
            self.schedulers = []
            for opt in self.model.get_opt_list():
                for g in opt.param_groups:
                    g['lr'] = self.lr_config['init']
                self.schedulers.append(ReduceLROnPlateau(opt, \
                    factor=self.lr_config['factor'], 
                    min_lr=self.lr_config['min_lr'], 
                    patience=self.lr_config['patience'], 
                    verbose=True, 
                    threshold=1e-5))
        elif self.scheduler == 'PlateauDecreaseScheduler':
            self.schedulers = [
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

        self.train_dataset = EnsembleDataset(
            self.dataset_path, self.label_dict, return_domain=True, dataset_type='train',device=self.device, emoca=self.emoca, debug=self.debug)
        self.valid_dataset = EnsembleDataset(
            self.dataset_path, self.label_dict, return_domain=False, dataset_type='valid',device=self.device, emoca=self.emoca, debug=self.debug)
        self.test_voca_dataset = BaselineVOCADataset(PATH['dataset']['vocoset'], device=self.device)

        trainer_sampler = RandomSampler(self.train_dataset)

        self.train_batchsize = trainer_conf['flexible_loader']['train_batchsize']
        self.train_loader = FlexibleLoader(self.train_dataset, 
                                           batchsize=trainer_conf['flexible_loader']['train_minibatch'], sampler=trainer_sampler, collate_fn=FACollate_fn)
        val_sampler = SequentialSampler(self.valid_dataset)
        self.valid_loader = FlexibleLoader(self.valid_dataset, 
                                           batch_size=trainer_conf['flexible_loader']['valid_batchsize'], sampler=val_sampler, collate_fn=FACollate_fn)

        # init output norm
        self.norm_dict = self.calculate_norm(self.train_dataset)
        self.model.set_norm(self.norm_dict['param_norm'], self.device)
        self.dan.set_norm(self.norm_dict['dan_norm'])
    
        # init criterion
        self.cri_vert = ParamLossFunc()
        self.cri_vert.set_hist(self.norm_dict['param_hist_list'])
        self.cri_pred = EmoTensorPredFunc()
        self.cri_pred.set_hist(self.norm_dict['dan_hist_list'])
        self.cri_mouth = MouthConsistencyFunc()
        self.cri_mouth.set_hist(self.norm_dict['param_hist_list'])

        # init queue for preloading
        self.queue_stream = Queue()
        self.queue = Queue()

        self.registered_loss = {'out_loss' : 0} # add loss item from out_dict in get_loss_item()

        # TODO complete grad check
        # self.plot_folder = '/home/chenyutong/facialanimation/figures'
        # self.gradcheck = GradCheck(self.model, (self.model_name + '_' + str(self.batch_size)) if self.debug == 0 else self.# model_name + '-debug', plot=True, plot_folder=self.plot_folder)
        
        os.makedirs(self.save_path, exist_ok=True)

    def calculate_norm(self, dataset):
        output = {}
        # load or create norm dict for dataset
        if os.path.exists(PATH['dataset']['norm_dict']):
            print('load from', PATH['dataset']['norm_dict'])
            sd_dataset = torch.load(PATH['dataset']['norm_dict'])
            param_norm = sd_dataset['norm']
            param_hist_list = sd_dataset['hist_list']
        else:
            params = []
            for idx in range(len(dataset)):
                data = dataset[idx]
                params.append(data['params'].to('cpu'))
            params = torch.cat(params, dim=0)
            param_norm = {'max' : torch.max(data['params'], dim=0).values, 'min' : torch.min(data['params'], dim=0).values}
            param_hist_list = cal_hist(
                params.permute(1,0), bins=15, save_fig=True, name_list=['param_' + str(idx) for idx in range(56)])
            torch.save({'norm':param_norm, 'hist_list':param_hist_list}, PATH['dataset']['norm_dict'])
        output['param_norm'] = param_norm
        output['param_hist_list'] = param_hist_list
        #param_norm['min'][53] = 0 # jaw angle

        # load or create norm dict for DAN Model
        print('max: ', param_norm['max'].max(dim=0).values, 'min: ', param_norm['min'].min(dim=0).values)
        if os.path.exists(PATH['3rd']['dan']['norm']):
            print('load from', PATH['3rd']['dan']['norm'])
            sd = torch.load(PATH['3rd']['dan']['norm'])
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
            torch.save({'norm':dan_out_norm, 'hist_list':dan_hist_list}, PATH['3rd']['dan']['norm'])
        print('dan norm:', dan_out_norm)
        output['dan_norm'] = dan_out_norm
        output['dan_hist_list'] = dan_hist_list
        return output

    def save_model(self):
        epoch = self.now_epoch
        # generate name by date and time
        today = date.today()
        da = today.strftime("%b-%d")
        now = datetime.now()
        current_time = now.strftime("%H-%M")
        filename = 'date-' + da + '-time-' + current_time + '-epoch-' + str(epoch) + '.pth'
        torch.save({
            'model' : self.model.state_dict(),
            'epoch' : self.epoch,
            'model-config' : GBL_CONF['model'][self.model_name],
            'trainer-config': GBL_CONF['trainer'],
            'norm' : self.norm_dict
            }, path.join(self.save_path, filename))
        print('model saved as ', filename)
        return path.join(self.save_path, filename)

    def load_model(self, model_path, emoca, device):
        sd, Model = load_model_dict(model_path, device)
        self.now_epoch = sd['epoch']
        self.model = Model.from_configs(sd['model-config'])
        self.model.load_state_dict(sd['model'])
        self.model.to(device=device)
        self.model.set_emoca(emoca)
        return self.model, self.now_epoch

    def get_loss_item(self, outs, gt):
        out_dict = {}
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
        if 'imgs' in data.keys():
            data['imgs'] = torch.cat(data['imgs'], dim=0)
            if 'emo_tensor' not in data.keys():
                data['emo_tensor'] = self.dan.inference(convert_img(data['imgs'], 'store', 'dan')).detach().to(self.device) # batch, max_seq_len, 7
        data['verts'] = self.emoca.decode(data['code_dict'], {'verts'}, target_device=self.device)['verts']
        
        if self.debug == 2: # single step
            save_img(convert_img(data['imgs'][0][0,...], i_code='store', o_code='tvsave'), 'ori.png')
        #data.pop('imgs')
        queue.put(data)

        return

    '''test phase (TODO add validation phase)'''
    def _test(self):
        test_avg_loss, test_max_loss = 0, 0
        lmk_idx = get_mouth_landmark('flame')
        templates = {}

        self.model = self.model.eval()
        with torch.no_grad():
            for idx, test_data in enumerate(self.test_voca_dataset):
                d = {
                    'wav':test_data['wav'].to(self.device), 
                    'code_dict':None , 
                    'name':test_data['name'], 
                    'seqs_len':test_data['seqs_len'], 
                    'flame_template':test_data['flame_template'], 
                    'shapecode':test_data['shapecode'].to(self.device)
                }
                d['emo_tensor_conf'] = ['no_use']
                gt = test_data['verts']
                seq_len = gt.size(0)
                params =  self.model.test_forward(d)['params'].squeeze(0) # 1, vertexm 3
                codedict = {'shapecode':torch.zeros((params.size(0), 100), device=self.device), 'expcode':params[:,:50], 'posecode':params[:,50:]}
                codedict['shapecode'] = test_data['shapecode'].to(self.device)
                output = self.flame.forward(codedict)
                output = torch.nn.functional.interpolate(output.unsqueeze(0).permute(0,2,1,3), size=(gt.size(0),3)).permute(0,2,1,3).squeeze(0)
                
                gt, output = gt.detach().cpu().numpy(), output.detach().cpu().numpy()
                '''
                model output is generated from FLAME blendshape, whcih is not exactly as same as FLAME template. 
                Transfer position delta to FLAME template can reduce unnecessary error in test loss
                '''
                output = np.asarray(output)
                if d['flame_template']['ply'] not in templates.keys():
                    tmp = PlyData.read(d['flame_template']['ply']) # load template dynamically
                    templates[d['flame_template']['ply']] = Mesh(vertices2nparray(tmp['vertex']), 'flame')
                output = output + (templates[d['flame_template']['ply']].v - output[0,:,:])

                seq_max_loss = 0
                seq_avg_loss = 0
                # TODO combine test code with baseline test
                for idx2 in range(seq_len):
                    m_out = Mesh(output[idx2,:,:], 'flame')
                    m_gt = Mesh(gt[idx2,:,:], 'flame')
                    m_out = approx_transform_mouth(m_out, m_gt)
                    delta = m_gt.v[lmk_idx,:]-m_out.v[lmk_idx,:]
                    delta = np.sqrt(np.power(delta[:,0],2) + np.power(delta[:,1],2) + np.power(delta[:,2],2))
                    seq_avg_loss += np.mean(delta)
                    seq_max_loss += np.max(delta)

                test_max_loss += (seq_max_loss / seq_len)
                test_avg_loss += (seq_avg_loss / seq_len)
                if idx % ceil(len(self.test_voca_dataset)/10) == 0:
                    print('idx=',idx, 'mean loss=', test_avg_loss/(idx+1), 'max loss=', test_max_loss/(idx+1), 'name=', test_data['name'])
        test_max_loss = test_max_loss / len(self.test_voca_dataset)
        test_avg_loss = test_avg_loss / len(self.test_voca_dataset)
        print(f'max loss={test_max_loss}, avg_loss={test_avg_loss}')
        return test_avg_loss, test_max_loss

    '''training phase'''
    def _train(self):
        self.model = self.model.train()
        for key in self.registered_loss.keys():
            self.registered_loss[key] = 0
        
        # init dataloader
        idx_it = 0
        it = iter(self.train_loader)
        transmit_cuda = {'emo_label', 'wav', 'params', 'emo_tensor'} # do not convert imgs into gpu
        
        self.queue.put(it)
        t_preload = Thread(target=self.preload, args=(self.device, transmit_cuda, self.queue))
        t_preload.start()
        t_preload.join()
        train_data = self.queue.get()

        while train_data is not None:
            for opt in self.model.get_opt_list():
                opt.zero_grad(set_to_none=True)
            self.queue.put(it)
            t_preload = Thread(target=self.preload, args=(self.device, transmit_cuda, self.queue))
            t_preload.start()
            
            out_dict = self.model.batch_forward(train_data) # forward

            out_dict['verts'] = self.emoca.decode(out_dict['code_dict'], {'verts'}, target_device=self.device)['verts']
            
            loss, loss_dict = self.get_loss_item(out_dict, train_data)

            for key in loss_dict.keys():
                if self.registered_loss.get(key) is not None: # first iteration only
                    self.registered_loss[key] += loss_dict[key]

            # backward
            loss.backward()
            # if self.epoch % 2 == 0 and idx_it == 0: # execute every 5 epoch
            #     self.gradcheck.check_grad(disp=True)

            # optimize
            if (self.train_loader.batch_size*(idx_it+1)) % self.train_batchsize == 0:
                for opt in self.model.get_opt_list():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    self.schedulers[0].step(self.epoch)
            del loss
            
            t_preload.join() # preload
            train_data = self.queue.get()
            idx_it += 1
            if len(self.train_loader) >= 5 and idx_it % round(len(self.train_loader)/5) == 0 and idx_it != 0: # execute 5 times per epoch
                print('iter ', idx_it, [key + '=' + str(self.registered_loss[key] / (idx_it+1)) for key in self.registered_loss.keys()])
        print('registered loss:', [key + '=' + "{:.3f}".format(self.registered_loss[key] / len(self.train_loader)) for key in self.registered_loss.keys()])

    def run_epoch(self):
        print('=======epoch: ', self.now_epoch, '=======')
        if self.lr_config['sche_type'] == 'PlateauDecreaseScheduler':
            print('lr=', self.schedulers[0].get_lr(), 'warmup step=', self.schedulers[0].get_wmp_step())
        else:
            print('lr=', self.model.get_opt_list()[0].param_groups[0]['lr'])
        
        test_avg_loss, test_max_loss = self._test()
        self._train()
        
        self.now_epoch += 1
        return test_max_loss
    
    def run_epochs(self):
        for _ in range(self.now_epoch, self.total_epoch, 1):
            self.run_epoch()
        
        
