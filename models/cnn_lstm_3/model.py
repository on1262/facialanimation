from transformers import Wav2Vec2Model, Wav2Vec2Config
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim

'''
version 3
use all lstm outputs to predict a whole sequence
'''

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

    def forward(self, x):
        x = self.norm(self.relu(self.den1(torch.reshape(x, (x.size(0), -1)))))
        return self.den2(x)
        

class OutputLSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(OutputLSTMLayer, self).__init__()
        self.lstm = nn.LSTM((in_channels), (128+out_channels), batch_first=True, proj_size=out_channels)

    def forward(self, x):
        out,_ = self.lstm(x)
        return out # not the last output


class Model(nn.Module):
    def __init__(
        self,
        emotion_class,
        emo_channels,
        act_channels,
        params_channels,
        seq_bottleneck,
        wav2vec_path,
        lstm_stack=1,
        dp=0,
        out_norm=None,
        debug=0
    ):
        super(Model, self).__init__()

        self.config = [emotion_class, emo_channels, act_channels, params_channels, seq_bottleneck, wav2vec_path, lstm_stack, dp, out_norm, debug]
        self.out_norm = out_norm
        try:
            model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, \
                config=join(wav2vec_path,'config.json'), state_dict=torch.load(join(wav2vec_path,'pytorch_model.bin')))
        except Exception as e:
            print('ignore wav2vec2 warning')
        self.fea_extractor = model.feature_extractor
        self.wav_layer = WavLayer(512, 64)
        self.out_layer = OutputLSTMLayer(64, 56)
        # opts
        self.opt_gen = optim.Adam(list(self.wav_layer.parameters()) + list(self.out_layer.parameters()))
        
        self.debug = debug
        self.wav_len_per_seq=16000/30.0
        self._norm_check = False

    @classmethod
    def from_configs(cls, config):
        return cls(*tuple(config))

    def get_configs(self):
        # update out norm
        self.config[-2] = self.out_norm
        return self.config

    def get_opt_list(self):
        return [self.opt_gen]

    def get_loss_mask(self):
        mask = torch.ones((56))*0.2
        mask[53] = 10 # jaw angle
        mask[[50,51,52]] = 0 # head pose
        return mask

    def set_norm(self, norm_dict, device):
        self.out_norm = norm_dict

        self.out_norm['min'] = self.out_norm['min'].to(device)
        self.out_norm['max'] = self.out_norm['max'].to(device)




    def batch_forward(self, wavs, seqs, seqs_len):
        params_num = torch.max(seqs_len)
        out_seqs = self.forward(wavs, seqs_len)
        return out_seqs
        
    def test_forward(self, wavs, seqs, seqs_len):
        return self.batch_forward(wavs, seqs, seqs_len)

    def forward(self, wavs, seqs_len):
        # input check
        assert(wavs.dim() == 2)
        batch_size = wavs.size(0)
        seq_length = torch.max(seqs_len).item()
        clip_len = round(self.wav_len_per_seq)
        clip_start = round(-1*self.wav_len_per_seq)
        clip_end = round(seq_length*self.wav_len_per_seq)
        if clip_end >= wavs.size(1):
            wavs = zero_padding(wavs, clip_end, dim=1)
        #print('pad end:', wavs.size(1))
        if clip_start < 0:
            wavs = zero_padding(wavs, -clip_start+wavs.size(1)+10, dim=1, t_first=False)
        #print('pad start:', wavs.size(1))
        #print('seq len', seq_length)
        wavs_input = None
        assert((seq_length+1)*clip_len < wavs.size(1))
        for tick in range(seq_length):
            if wavs_input is None:
                wavs_input = wavs[:,0:(tick+2)*clip_len].unsqueeze(dim=1)
            else:
                wavs_input = torch.cat([wavs_input, wavs[:,tick*clip_len:(tick+2)*clip_len].unsqueeze(dim=1)], dim=1)
        
        wav_out = None
        for tick in range(seq_length):
            wav_fea = self.fea_extractor(wavs_input[:,tick,:]).transpose(-1,-2) # batch_size, fea_len, 512
            if wav_out is None:
                wav_out = self.wav_layer(wav_fea).unsqueeze(1) # batch, 1, out_fea
            else:
                wav_out = torch.cat([wav_out, self.wav_layer(wav_fea).unsqueeze(1)], dim=1)

        
        
        # calculate parameters
        
        params = self.out_layer(wav_out)
        

        #if self.debug == 1:
        #    print('wavs:', wavs.size())         # [1, len]
        #    print('wavs_input:', wavs_input.size())         # [1, len]
        #    print('wav_fea', wav_fea.size())  # [1, 13, fea_len, 768]
        #    print('wav_out', wav_out.size())
        #    print('params ', params.size())   # [56]
        
        #return dis_out.expand({'params':params}) # dis_in1, dis_in2, params
        if self.out_norm is not None:
            if self._norm_check == False or self.out_norm['min'].device != wavs.device:
                dev = wavs.device
                self.out_norm['min'] = self.out_norm['min'].to(dev)
                self.out_norm['max'] = self.out_norm['max'].to(dev)
                self._norm_check = True
            params =  self.out_norm['min'] + (params+1) / 2 * (self.out_norm['max'] - self.out_norm['min']) 
        return params

