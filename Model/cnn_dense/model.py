from transformers import Wav2Vec2Model, Wav2Vec2Config
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim

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

class LSTMExtractorLayer(nn.Module):
    def __init__(self, output_channels, num_stack=1,dp=0.1):
        super(LSTMExtractorLayer, self).__init__()
        #self.avg_mat = nn.Parameter(0.1*torch.ones((13,768)),requires_grad=True)
        self.dp = nn.Dropout(dp)
        self.lstm = nn.LSTM((512), (256+output_channels), proj_size=output_channels, batch_first=True, num_layers=num_stack)

    def forward(self, hidden:torch.Tensor):
        # hidden: (batch_size, sequence_length, hidden_size)
        # label predictor
        x = self.dp(hidden)
        x,_ = self.lstm(x) #(N,L,output_channels)
        x = x[:,-1,:] # (output_channels), batch_size first
        return x

class ExtractorLayer(nn.Module):
    def __init__(self, output_channels,dp=0.1):
        super(ExtractorLayer, self).__init__()
        self.out_channels = output_channels
        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_channels),
            nn.ReLU()
        )

    def forward(self, hidden):
        return self.layer(hidden)

class LSTMOutputLayer(nn.Module):
    def __init__(self, t0_channels, seq_channels, seq_bottleneck, out_channels,num_stack,dp=0.1):
        super(LSTMOutputLayer, self).__init__()
        self.out_channels = out_channels
        self.den1 = nn.Linear(seq_channels,seq_bottleneck)
        self.lstm = nn.LSTM(input_size=(seq_bottleneck+t0_channels), hidden_size=2*out_channels, batch_first=True, num_layers=num_stack, \
            proj_size=out_channels, dropout=0 if num_stack == 1 else dp)

    def forward(self, t0, seq):
        seq_bn = self.den1(seq)
        # print(seq_bn.size(), t0.size())
        out,_ = self.lstm(torch.cat([seq_bn, t0.unsqueeze(dim=1).repeat(1,seq_bn.size(1),1)],dim=2))
        return out

class OutputLayer(nn.Module):
    def __init__(self, t0_channels, out_channels,dp=0.1):
        super(OutputLayer, self).__init__()
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Linear(t0_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

    def forward(self, t0, seq):
        return self.layer(t0)

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
        model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, \
            config=join(wav2vec_path,'config.json'), state_dict=torch.load(join(wav2vec_path,'pytorch_model.bin')))
        self.fea_extractor = model.feature_extractor
        self.emotionLayer = ExtractorLayer(output_channels=emo_channels, dp=dp)
        self.actionLayer = ExtractorLayer(output_channels=act_channels, dp=dp)
        #self.discriminator = Discriminator(emotion_class=emotion_class, in1_channels=emo_channels, in2_channels=act_channels,dp=dp)
        self.outLayer = OutputLayer(
            t0_channels=emo_channels+act_channels,
            out_channels=params_channels,
            dp=dp
            )
        # opts
        self.opt_tf = optim.Adam(self.fea_extractor.parameters())
        self.opt_gen = optim.Adam(list(self.actionLayer.parameters()) \
            + list(self.emotionLayer.parameters()) \
            + list(self.outLayer.parameters()))
        
        self.debug = debug
        self.wav_len_per_seq=16000/30.0
        self._norm_check = False
        self.sig = nn.Sigmoid()

    @classmethod
    def from_configs(cls, config):
        return cls(*tuple(config))

    def get_configs(self):
        # update out norm
        self.config[-2] = self.out_norm
        return self.config

    def get_opt_list(self):
        return [self.opt_tf, self.opt_gen]

    def set_norm(self, norm_dict, device):
        self.out_norm = norm_dict
        #delta = (self.out_norm['max'] - self.out_norm['min'])/2.0 # let x in -0.5~0.5, y=sigmoid(x) in min-delta~max+delta
        #self.out_norm['max'] += delta
        #self.out_norm['min'] -= delta

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
        #if self.debug == 1:
        #    print('in_list:', in_list)
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
            #if self.debug == 1:
            #    print(torch.sum(out_tensor, dim=2))
        return out_tensor
        

    def forward(self, wavs, seqs):
        # input check
        assert(wavs.dim() == 2 and seqs.dim() == 3)
        assert(wavs.size(0) == seqs.size(0))
        batch_size = wavs.size(0)
        tick = seqs.size(-2) - 1
        wav_up = round((tick+2)*self.wav_len_per_seq)
        wav_down = max(0, round((tick)*self.wav_len_per_seq))
        #if self.debug == 1:
        #    print('wav up=', wav_up, 'wav_down=', wav_down, ' delta=', wav_up-wav_down)
        if wav_up >= wavs.size(1):
            #if wav_up - wavs.size(1) > 5:
            #    print('warning, wav_up=', wav_up, ' wavs max len=', wavs.size(1))
            wavs = zero_padding(wavs, wav_up, dim=1)
        
        wavs = wavs[:, wav_down:wav_up]
        # add 10ms zero padding to make sure wav2vec2 works well
        wavs = torch.cat([wavs, torch.zeros((batch_size, 160), device=wavs.device)], dim=1)
        wav_fea = None
        try:
            wav_fea = self.fea_extractor(wavs).transpose(1,2) # batch_size, aud_seq_len, 512
            wav_fea = torch.mean(wav_fea, dim=1)
        except Exception as e:
            print('error occured, wavs size:' , wavs.size())
            
        #wav_fea = torch.stack(wav_fea, dim=1) 
        
        # emotion feature
        emo_fea = self.emotionLayer(wav_fea)
        
        # action feature
        act_fea = self.actionLayer(wav_fea)
        
        # calculate parameters
        params = self.outLayer(t0=torch.cat([emo_fea, act_fea], dim=-1), seq=seqs)
        
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

