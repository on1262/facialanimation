import torch.nn as nn

from utils.balance_data import hist_inv_func_dict
from utils.config_loader import PATH
import torch
import os, json


def get_front_face_idx():
    with open(os.path.join(PATH['fitting'], 'front-face-idx.json'), 'r') as f:
        idx_list = json.load(f)['front_face']
    return idx_list


class ParamLossFunc():
    def __init__(self):
        self.l1loss = nn.SmoothL1Loss(reduction='none', beta=0.01)
        # generate loss mask
        self.loss_mask = torch.ones(5023,3)
        self.hist = None
        self.idx_mask = get_front_face_idx()

    def set_hist(self, hist_list:list):
        self.hist = hist_list
    
    def cal_loss(self, seq_eval, seq_gt, out_mask=None, use_hist=False):
        assert(seq_eval.size() == seq_gt.size())
        ori_size = seq_gt.size()
        seq_gt = seq_gt.to(seq_eval.device)
        self.loss_mask = self.loss_mask.to(seq_eval.device)
        seq_eval, seq_gt = torch.mul(seq_eval, self.loss_mask), torch.mul(seq_gt, self.loss_mask)
        
        assert(seq_eval.size() == ori_size and seq_gt.size() == ori_size)

        if use_hist:
            hist_loss_mask = seq_gt.detach().clone()
            for b in range(hist_loss_mask.size(0)):
                for idx in range(hist_loss_mask.size(-1)):
                    hist_loss_mask[b,:,idx] = hist_inv_func_dict(self.hist[idx], hist_loss_mask[b,:,idx])

        #vol_loss = torch.mean(torch.abs(torch.diff(torch.mean(seq_eval - seq_gt, dim=-1), dim=-1))) * seq_gt.size(1)
        #if use_hist:
        #    pos_loss =  (self.l1loss(seq_eval, seq_gt)*hist_loss_mask).mean() * seq_gt.size(1)
        #else:
            # pos_loss =  (self.l1loss(seq_eval, seq_gt)).mean() * seq_gt.size(1)
        pos_loss =  (self.l1loss(seq_eval[:,self.idx_mask, :], seq_gt[:,self.idx_mask, :])).mean() * 50
        
        #loss = pos_loss + 5*vol_loss

        return pos_loss

class MouthConsistencyFunc():
    def __init__(self):
        # key param in 50 expression params
        self.mouth_weights = []
        '''
        upper lip: 3547
        left corner: 2845, right corner: 1730
        bottom lip: 3513
        '''
        self.hist = None
        self.lip_vertex = [3547, 3513, 2845, 1730]
        self.mouth_v = [1, 3, 5]
        self.mouth_h = [0, 1, 3, 6, 7] # idx 0 is scaled up
        self.params_range = {0:2.0, 1:10.0, 3:6.0, 5:7.3, 6:10.4, 7:7.0, 53:0.75}
        # loss mask of lip movement
        self.loss_mask = torch.zeros(56)
        for idx in range(56):
            if idx in self.mouth_h or idx in self.mouth_v or idx == 53:
                self.loss_mask[idx] = 1.0 / self.params_range[idx]
        self.lip_coeff = [1.0/0.017, 1.0/0.0476] # vertical, horizontal

        self.func = nn.L1Loss(reduction='none')
        self.relu = torch.nn.ReLU()
    
    def set_hist(self, hist):
        self.hist = hist

    def cal_loss(self, out_params, gt_params, out_verts, gt_verts, gt_wavs, out_mask=None, use_hist=False):
        try:
            assert(out_params.device == gt_params.device and out_verts.device == gt_verts.device)
            assert(out_verts.dim() == 3 and out_params.dim() == 3)
            assert(out_verts.size(0) == gt_verts.size(0))
        except AssertionError as e:
            # TODO
            print('assert error')
            print('outverts', out_verts.size(), 'gt verts', gt_verts.size())
            print('out params', out_params.size(), 'gt params', gt_params.size())
        out_verts = out_verts[:, self.lip_vertex, :] # sum(seq_len), 4, 3
        gt_verts = gt_verts[:out_verts.size(0), self.lip_vertex, :]
        if out_mask is not None:
            out_params, gt_params = torch.mul(out_params, out_mask), torch.mul(gt_params, out_mask)
        if self.loss_mask.device != out_params.device:
            self.loss_mask = self.loss_mask.to(out_params.device)
        out_params, gt_params = torch.mul(out_params, self.loss_mask), torch.mul(gt_params, self.loss_mask)

        if use_hist:
            hist_loss_mask = gt_params.detach().clone()
            for b in range(hist_loss_mask.size(0)):
                for idx in [0,1,3,5,6,7,53]:
                    hist_loss_mask[b,:,idx] = hist_inv_func_dict(self.hist[idx], hist_loss_mask[b,:,idx])
        # cal mute clip
        mute_mask = torch.ones((out_params.size(0), out_params.size(1)), device=out_params.device)
        thres = torch.mean(torch.abs(gt_wavs), dim=1) * 0.35
        for i in range(out_params.size(1)):
            w_start = round(i* 16000/30)
            w_end = round(min(gt_wavs.size(1), (i+1) * 16000/30))
            mute_mask[:,i] *= (torch.mean(torch.abs(gt_wavs[:, w_start:w_end]), dim=1) < thres)
        for i in range(1, out_params.size(1)-1):
            mute_mask[:,i] *= (mute_mask[:,i] + mute_mask[:, i-1] + mute_mask[:, i+1]) > 1.5
        # print('mute mask:', mute_mask.mean(dim=1))
        # for i in range(10):
        #     print(f'mute mask [{i}]:', mute_mask[i,:])
        jaw_loss = self.func(out_params[:,:,53], gt_params[:,:,53])
        
        jaw_vol_loss = torch.mean(torch.abs(torch.diff(jaw_loss, dim=-1))) * gt_params.size(1)

        jaw_loss = torch.abs((mute_mask * torch.abs(out_params[:,:,53])).mean())

        if use_hist:
            param_loss = (self.func(out_params, gt_params) * hist_loss_mask).mean()
        else:
            param_loss = (self.func(out_params, gt_params)).mean()

        mouth_v = ((out_verts[:,0,:] - out_verts[:,1,:]) \
            - (gt_verts[:,0,:] - gt_verts[:,1,:])) * self.lip_coeff[0]
        
        mouth_h = ((out_verts[:,3,:] - out_verts[:,2,:]) \
            - (gt_verts[:,3,:] - gt_verts[:,2,:])) * self.lip_coeff[1]
        
        mouth_loss = torch.sum((torch.abs(mouth_v) + torch.abs(mouth_h))) / (out_params.size(0)) # no need to * 3 because loss is only in 1 dim

        return jaw_loss * 250, jaw_vol_loss, param_loss * 10, mouth_loss * 0.5


class EmoTensorPredFunc():
    def __init__(self):
        self.hist = None
    
    def smooth(self, x):
        # x: seq_len.sum(batch), 7
        return torch.nn.functional.conv1d(
            x.unsqueeze(0).permute(0,2,1), torch.ones((7,1,31), device=x.device)/31, padding=15, groups=7).permute(0,2,1).squeeze(0)

    def set_hist(self, hist_list:list):
        self.hist = hist_list

    def cal_loss(self, out_emo_tensor:torch.Tensor, gt_emo_tensor:torch.Tensor, seqs_len):
        tmp = gt_emo_tensor.clone()
        for idx in range(seqs_len.size(0)):
            start = 0 if idx == 0 else seqs_len[:idx].sum()
            tmp[start:start+seqs_len[idx],:] = self.smooth(gt_emo_tensor[start:start+seqs_len[idx],:])
        return (torch.abs(out_emo_tensor - tmp)).mean()