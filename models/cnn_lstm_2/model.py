from transformers import Wav2Vec2Model, Wav2Vec2Config
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim

'''
auto-regression
predict 53 parameters
wav_fea->LSTM
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



class LSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM((in_channels), (128), batch_first=True, proj_size=out_channels)

    def forward(self, x):
        out,_ = self.lstm(x)
        return out[:,-1,:]

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
        self.lstm_layer = LSTMLayer(512, 56)
        # opts
        self.opt_gen = optim.Adam(list(self.lstm_layer.parameters()))
        
        self.debug = debug
        self.wav_len_per_seq=16000/30.0
        self._norm_check = False
        self.sig = nn.Hardsigmoid() #

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

    def seqs_zero_init(self, wavs, seqs):
        batch_size = wavs.size(0)

        # insert init frame to seqs
        if seqs is not None:
            seqs = torch.cat([torch.zeros((batch_size, 1, seqs.size(-1)), device=wavs.device), seqs], dim=-2)
        else:
            seqs = torch.zeros((wavs.size(0), 1, 56), device=wavs.device)
        return seqs

    def train_process(self, wavs, seqs, seqs_len, tick): # seq should be initialized first
        # seq: zero, 0, 1, ... ,N tick=N-1->N
        assert(tick < torch.max(seqs_len))
        assert(seqs is not None and seqs.size(0) == wavs.size(0))
        in_list = [idx for idx in range(seqs_len.size(0)) if seqs_len[idx] > tick]
        wavs = wavs[in_list,:]

        seqs_gt = seqs[in_list, tick+1,:]
        seqs = seqs[in_list,:tick+1,:] # [zero, 0, 1, 2,...., tick-1] length=tick+1
        return wavs, seqs, seqs_gt, in_list

    def batch_forward(self, wavs, seqs, seqs_len):
        params_num = seqs.size(-2)
        assert(torch.max(seqs_len) == seqs.size(-2)) # seq_len is origin seq
        #max_seq_len = torch.max(seqs_len)
        assert(seqs.dim() == 3)
        out_tensor = torch.zeros(seqs.size(), device=seqs.device)
        seqs = self.seqs_zero_init(wavs, seqs)
        
        for tick in range(params_num):
            #tick = random.choice(range(params_num))
            wav_iter, seqs_iter, seq_gt, in_list = self.train_process(wavs, seqs, seqs_len, tick)
            # forward
            out = self.forward(wavs=wav_iter, seqs=seqs_iter) # 1->tick-1
            out_tensor[in_list,tick,:] += out
        return out_tensor
        
    def test_forward(self, wavs, seqs, seqs_len):
        params_num = torch.max(seqs_len)
        assert(seqs is None and seqs_len.size(0) == 1) # batch size is 1
        out_tensor = torch.zeros((1, params_num, 56), device=wavs.device)
        seqs = self.seqs_zero_init(wavs, seqs)

        for tick in range(params_num):
            # forward
            out = self.forward(wavs=wavs, seqs=seqs) # 1->tick-1
            seqs = torch.cat([seqs, out.unsqueeze(1)], dim=1)
        return seqs[:,1:,:] # remove zero init


    def forward(self, wavs, seqs):
        # input check
        assert(wavs.dim() == 2 and seqs.dim() == 3)
        assert(wavs.size(0) == seqs.size(0))
        batch_size = wavs.size(0)
        tick = seqs.size(-2) - 1
        wav_up = round((tick+1.5)*self.wav_len_per_seq) # tick+x, x=wav_fea_len*333/533, 20ms per wav fea
        wav_down = round((tick-1.5)*self.wav_len_per_seq)
        if wav_up >= wavs.size(1):
            wavs = zero_padding(wavs, wav_up, dim=1)
        
        if wav_down < 0:
            wavs = zero_padding(wavs, -wav_down+wavs.size(1),dim=1, t_first=False)
            wavs = wavs[:, 0:wav_up-wav_down]
        else:
            wavs = wavs[:, wav_down:wav_up]
        wav_fea = None
        try:
            wav_fea = self.fea_extractor(wavs).transpose(1,2) # batch_size, aud_seq_len, 512
        except Exception as e:
            print('error occured, wavs size:' , wavs.size())
        
        # calculate parameters
        params = self.lstm_layer(wav_fea)
        
        # Discriminator
        #dis_out = self.discriminator(in1=emo_fea, in2=act_fea)

        #if self.debug == 1:
        #    print('wavs:', wavs.size())         # [1, len]
        #    print('emo_fea ', emo_fea.size()) # [1, emo_feature] emo_feature=128
        #    print('wav_fea', wav_fea.size())  # [1, 13, fea_len, 768]
        #    print('act_fea ', act_fea.size()) # [1, act_fea] act_fea=128
        #    print('params ', params.size())   # [56]
        
        #return dis_out.expand({'params':params}) # dis_in1, dis_in2, params
        if self.out_norm is not None:
            if self._norm_check == False or self.out_norm['min'].device != wavs.device:
                dev = wavs.device
                self.out_norm['min'] = self.out_norm['min'].to(dev)
                self.out_norm['max'] = self.out_norm['max'].to(dev)
                self._norm_check = True
            params =  self.out_norm['min'] + self.sig(params) * (self.out_norm['max'] - self.out_norm['min']) 
        return params

