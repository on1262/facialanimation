import sys
sys.path.append('/home/chenyutong/facialanimation')
from utils.interface import EMOCAModel, GSTModel
from transformers import Wav2Vec2Model, Wav2Vec2Config
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def apply_V_mask(lstm_out: torch.Tensor):
    mask = torch.linspace(0, lstm_out.size(1), steps=lstm_out.size(1), device=lstm_out.device, requires_grad=False) / lstm_out.size(1)
    (b,_,_,p) = lstm_out.size()
    mask = mask.view(1,-1,1).expand(b,-1,p) # batch, seqs_len, 1, params
    return lstm_out[:,:,0,:] * mask + lstm_out[:,:,1,:] * (1.0 - mask)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def zero_padding(t:torch.tensor, pad_size, dim=-1, t_first=True):
    assert(t.size(dim) <= pad_size)
    dev = t.device
    if t.size(dim) == pad_size:
        return t
    if t.dim() == 1:
        if t_first:
            return torch.cat([t, torch.zeros((pad_size-t.size(0)), device=dev)])
        else:
            return torch.cat([torch.zeros((pad_size-t.size(0)), device=dev),t])
    else:
        p_size = list(t.size())
        p_size[dim] = pad_size - t.size(dim)
        p_size = torch.Size(p_size)
        if t_first:
            return torch.cat([t, torch.zeros(p_size, device=dev)], dim=dim)
        else:
            return torch.cat([torch.zeros(p_size, device=dev), t], dim=dim)

class WavLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(WavLayer, self).__init__()
        self.den1 = nn.Linear(in_channels*3, 256)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        self.den2 = nn.Linear(256, out_channels)
        self.dropout = nn.Dropout(dp)

    def forward(self, x):
        x = self.norm(self.relu(self.dropout(self.den1(torch.reshape(x, (x.size(0), -1))))))
        return self.den2(x)
        


class LSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM((in_channels), (64+out_channels), batch_first=True, bidirectional=True, proj_size=out_channels)
        #self.alpha = nn.Parameter(data=torch.Tensor([0.5]), requires_grad=True)
        self.dropout = nn.Dropout(dp)

    def forward(self, x):

        out,_ = self.lstm(self.dropout(x))
        out = out.view(out.size(0), out.size(1), 2, out.size(2) // 2)
        out = apply_V_mask(out)
        return out # not the last output

class EmoPredLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(EmoPredLayer, self).__init__()
        self.c_emo = out_channels
        self.lstm = nn.LSTM((in_channels), (128), batch_first=True, bidirectional=True, proj_size=out_channels)
        self.dropout = nn.Dropout(dp)
        # manually norm
        self.emo_min = -2
        self.emo_max = 3

    def forward(self, x, seq_len):

        out,_ = self.lstm(self.dropout(x))
        out = out.view(out.size(0), out.size(1), 2, out.size(2) // 2)
        out = apply_V_mask(out)
        out = (out + 1) * 0.5 * (self.emo_max - self.emo_min) + self.emo_min
        # parse
        out_emo_tensor = torch.zeros(seq_len.sum(), self.c_emo, device=x.device)
        for idx in range(seq_len.size(0)):
            start = 0 if idx == 0 else seq_len[:idx].sum()
            out_emo_tensor[start:start+seq_len[idx],:] = out[idx, 0:seq_len[idx],:]
        return out_emo_tensor # not the last output

class StyleTokenLayer(nn.Module):
    """
    StyleTokenLayer
    input:
        emo_tensor: Batch, seq_len, emo_channel
    output:
        Batch, seq_len, out_hidden
    """
    def __init__(self, emo_channels, out_hidden=768, in_hidden=768, dp=0.1):
        super().__init__()
        self.c_emo = emo_channels
        self.hid = in_hidden
        self.style_embedding = nn.Parameter(torch.zeros((1,emo_channels, in_hidden)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.style_embedding)
        
    
    def forward(self, emo_tensor, seqs_len):
        #assert(emo_tensor.dim()==3)
        out_emo_tensor = torch.zeros(seqs_len.size(0), seqs_len.max(), self.c_emo, device=emo_tensor.device)
        for idx in range(seqs_len.size(0)):
            start = 0 if idx == 0 else seqs_len[:idx].sum()
            out_emo_tensor[idx, 0:seqs_len[idx],:] = emo_tensor[start:start+seqs_len[idx],:]
        ebd = self.style_embedding.expand(seqs_len.size(0), self.c_emo, self.hid).to(emo_tensor.device)
        return torch.bmm(out_emo_tensor, ebd)


class Model(nn.Module):
    def __init__(
        self,
        emo_channels,
        params_channels,
        wav2vec_path,
        dp=0,
        out_norm=None,
        debug=0
    ):
        super(Model, self).__init__()

        self.config = [emo_channels, params_channels, wav2vec_path, dp, out_norm, debug]
        self.out_norm = out_norm
        self.wav_fea_channels = 512
        self.params_channels = params_channels
        self.hidden = 256
        self.cp_pred = 25 # only predict first 25 expression params
        self.emo_channels = emo_channels
        # emoca
        self.emoca = None
        self.debug = debug
        self.wav_len_per_seq=16000/30.0
        self._norm_check = False

        try:
            model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, \
                config=join(wav2vec_path,'config.json'), state_dict=torch.load(join(wav2vec_path,'pytorch_model.bin')))
        except Exception as e:
            print('ignore wav2vec2 warning')
        self.fea_extractor = model.feature_extractor
        for p in self.fea_extractor.parameters(recurse=True): # freeze
            p.requires_grad = False
        self.wav_layer = WavLayer(self.wav_fea_channels, self.hidden)
        self.lstm_layer = LSTMLayer(self.hidden, self.cp_pred,dp=dp)
        self.pose_layer = LSTMLayer(self.hidden, 6, dp=dp)
        self.pred_layer = EmoPredLayer(self.hidden, self.emo_channels)
        self.ebd_convert = nn.Linear(512, self.hidden)
        #self.pred_layer.load_state_dict(torch.load('/home/chenyutong/facialanimation/Model/lstm_emo/pred_pretrained.pth'))
        #for p in self.pred_layer.parameters(recurse=True):
        #    p.requires_grad = False
        self.style_layer = StyleTokenLayer(emo_channels, out_hidden=self.hidden, in_hidden=self.hidden, dp=0.1)
        # opts
        self.opt_gen = optim.Adam(list(self.wav_layer.parameters(recurse=True))
            + list(self.lstm_layer.parameters(recurse=True))
            + list(self.pose_layer.parameters(recurse=True))
            + list(self.pred_layer.parameters(recurse=True))
            + list(self.ebd_convert.parameters())
        )
        self.opt_style = optim.Adam(list(self.style_layer.parameters(recurse=True)))
        

    def batch_forward(self, in_dict):
        return self.forward(
            wavs=in_dict['wav'], 
            seqs_len=in_dict['seqs_len'], 
            emo_tensor=in_dict['emo_tensor'], 
            code_dict=in_dict['code_dict'], 
            style_ebd=in_dict['style_ebd'])
        
    def test_forward(self, in_dict):
        assert(in_dict['wav'].size(0) == 1) # batch_size should be 1
        if in_dict.get('emo_tensor') is None:
            in_dict['emo_tensor'] = self.forward(in_dict['wav'], in_dict['seqs_len'], None,  pred_only=True)
            if in_dict.get('emo_label') is not None:
                labels = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
                convert_list = {0:0, 1:1, 3:2, 4:5, 5:4, 2:6}
                emo_index = convert_list[in_dict['emo_label'].item()]
                print('adjust emo label:', labels[emo_index])
                #zero_tensor = torch.zeros(in_dict['emo_tensor'].size(), device=in_dict['emo_tensor'].device)
                #zero_tensor += 2.0
                in_dict['emo_tensor'][:, emo_index] += 2.0
        out =  self.forward(
            wavs=in_dict['wav'], 
            seqs_len=in_dict['seqs_len'], 
            emo_tensor=in_dict['emo_tensor'], 
            code_dict=[in_dict['code_dict']], 
            style_ebd=in_dict['style_ebd'])
        return out
    
    def forward(self, wavs, seqs_len, emo_tensor=None, code_dict=None, style_ebd=None, pred_only=False):
        dev = wavs.device # regard as input device
        output = {}
        # input check
        assert(wavs.dim() == 2)
        if emo_tensor is not None:
            assert(wavs.device == emo_tensor.device)
        seq_length = torch.max(seqs_len).item()

        # generate mask
        output['mask'] = self.get_output_mask(seqs_len).to(dev)

        clip_len = round(self.wav_len_per_seq)
        if round(seq_length*self.wav_len_per_seq) >= wavs.size(1):
            wavs = zero_padding(wavs, round(seq_length*self.wav_len_per_seq), dim=1)
        wavs = zero_padding(wavs, round(self.wav_len_per_seq)+wavs.size(1), dim=1, t_first=False)

        wavs_input_list = []
        assert((seq_length+1)*clip_len < wavs.size(1))
        for tick in range(seq_length):
            wavs_input_list.append(wavs[:, tick*clip_len:(tick+2)*clip_len].clone())
        wavs_input = torch.stack(wavs_input_list, dim=1)
                
        wav_out = None
        for tick in range(seq_length):
            wav_fea = self.fea_extractor(wavs_input[:,tick,:]).transpose(-1,-2) # batch_size, fea_len, 512
            if wav_out is None:
                wav_out = self.wav_layer(wav_fea).unsqueeze(1) # batch, 1, out_fea
            else:
                wav_out = torch.cat([wav_out, self.wav_layer(wav_fea).unsqueeze(1)], dim=1) # batch, seqs_len, out_fea

        style_ebd = self.ebd_convert(style_ebd) # use style ebd to predict emotion tensor
        style_ebd = style_ebd.unsqueeze(1).expand(-1, wav_out.size(-2), -1) # batch, seqs_len, 512 
        pred = self.pred_layer(wav_out.detach() + style_ebd, seqs_len)

        if pred_only or emo_tensor is None:
            return pred
        else:
            output['pred_emo_tensor'] = pred
        
        emo_ebd = self.style_layer(emo_tensor, seqs_len).expand(wav_out.size(0),-1,-1)
        exp_out = self.lstm_layer(wav_out + emo_ebd + style_ebd)
        zero_fill = torch.zeros((exp_out.size(0), exp_out.size(1), self.params_channels-self.cp_pred-6)).to(dev)
        pose_out = self.pose_layer(wav_out.detach() + style_ebd)
        lstm_out = torch.cat([exp_out, zero_fill, pose_out], dim=-1)

        if self.out_norm is not None:
            if self._norm_check == False or self.out_norm['min'].device != dev:
                self.out_norm['min'] = self.out_norm['min'].to(dev)
                self.out_norm['max'] = self.out_norm['max'].to(dev)
                self._norm_check = True
            lstm_out = self.out_norm['min'] + (lstm_out+1) *0.5 * (self.out_norm['max'] - self.out_norm['min'])

        #output['params'] = lstm_out * output['mask'] # apply mask
        output['params'] = lstm_out * output['mask']
        
        if code_dict is not None:
            if self.emoca is None:
                self.emoca = EMOCAModel(device=dev)
            # replace code_dict
            for idx in range(len(code_dict)):
                try:
                    assert(code_dict[idx]['expcode'].size(0) == seqs_len[idx])
                except AssertionError as e:
                    print('codedict error', code_dict[idx]['expcode'].size(0), seqs_len[idx], seqs_len)
                # code_dict[idx]['expcode'] = lstm_out[idx, :seqs_len[idx], :50]
                code_dict[idx]['expcode'] = lstm_out[idx, :seqs_len[idx], :50]
                code_dict[idx]['posecode'][:,3] = lstm_out[idx, :seqs_len[idx], 53]
                code_dict[idx]['posecode'][:,[4,5]] = 0 # fixed dim

            output['code_dict'] = code_dict
            

        return output

    @classmethod
    def from_configs(cls, config):
        return cls(*tuple(config))

    def get_configs(self):
        # update out norm
        self.config[-2] = self.out_norm
        return self.config

    def get_opt_list(self):
        return [self.opt_gen, self.opt_style]

    def get_output_mask(self, seqs_len):
        max_seqs_len = max(seqs_len)
        # mask for seqs_len = tensor([3,2,5])
        # [[T,T,T,F,F],
        #  [T,T,F,F,F],
        # [T,T,T,T,T]]
        mask = torch.arange(0, max_seqs_len, 1).long().unsqueeze(0).expand(seqs_len.size(0), max_seqs_len) < seqs_len.unsqueeze(0).transpose(0,1)
        mask = mask.unsqueeze(dim=-1).expand(mask.size(0), mask.size(1), 56)
        return mask # no device

    def set_norm(self, norm_dict, device):
        self.out_norm = norm_dict
        self.out_norm['min'] = self.out_norm['min'].to(device)
        self.out_norm['max'] = self.out_norm['max'].to(device)

    def set_emoca(self, emoca):
        self.emoca = emoca
