from transformers import Wav2Vec2Model, Wav2Vec2Config
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

'''
bert based transformer decoder
pipeline: CNN->lstm->transformer decoder
version5: absolute position embedding
predict whole sequence
'''

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


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

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :x.size(1)]

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


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, mode='cross'):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.mode = mode
        if mode == 'self':
            self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
            self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, c, mask):
        if self.mode == 'cross' and c is not None:
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, c, c, mask=mask))
        else:
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
            x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

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
        

class LSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels,dp=0.1):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM((in_channels), (128+out_channels), batch_first=True, proj_size=out_channels)

    def forward(self, x):
        out,_ = self.lstm(x)
        return out # not the last output


class TransformerModel(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, n_params_fea=56, n_wav_fea=512, hidden=768, n_layers=2, attn_heads=6, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        self.input_linear = nn.Linear(n_params_fea, hidden)
        self.pos_ebd = PositionalEmbedding(hidden)
        self.out_linear = nn.Linear(hidden, n_params_fea)
        self.out_activ = nn.Tanh()
        self.wav_linear = nn.Linear(n_wav_fea, hidden)

        # multi-layers transformer blocks, deep network
        self.cross_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, mode='cross') for _ in range(n_layers)])
        
        self.self_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, mode='self') for _ in range(n_layers)])

    def forward(self, x, c, seqs_len):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        c = self.wav_linear(c)
        x = self.input_linear(x) #batch_size, seq_len, hidden
        for idx in range(x.size(0)):
            x[idx, :seqs_len[idx],:] += self.pos_ebd(x[idx, :seqs_len[idx],:])

        # running over multiple transformer blocks
        for idx in range(self.n_layers):
            x = self.self_transformer_blocks[idx].forward(x, None, mask=None)
            x = self.cross_transformer_blocks[idx].forward(x, c, mask=None)

        return self.out_activ(self.out_linear(x))

class Model(nn.Module):
    def __init__(
        self,
        params_channels,
        wav2vec_path,
        dp=0,
        out_norm=None,
        debug=0
    ):
        super(Model, self).__init__()

        self.config = [params_channels, wav2vec_path, dp, out_norm, debug]
        self.out_norm = out_norm
        self.wav_fea_channels = 512
        try:
            model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path=None, \
                config=join(wav2vec_path,'config.json'), state_dict=torch.load(join(wav2vec_path,'pytorch_model.bin')))
        except Exception as e:
            print('ignore wav2vec2 warning')
        self.fea_extractor = model.feature_extractor
        self.wav_layer = WavLayer(self.wav_fea_channels, 64)
        self.lstm_layer = LSTMLayer(64, 56)
        self.transformer_layer = TransformerModel(
            n_params_fea=params_channels, n_wav_fea=self.wav_fea_channels, hidden=768, n_layers=2, attn_heads=6, dropout=dp)
        # opts
        self.opt_gen = optim.Adam(list(self.wav_layer.parameters()) 
            + list(self.lstm_layer.parameters()) 
            + list(self.transformer_layer.parameters()))
        
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
        seq_length = torch.max(seqs_len).item()

        clip_len = round(self.wav_len_per_seq)
        cond_wav_fea = self.fea_extractor(wavs).transpose(-1,-2) # calculate wav feature in whole sequence
        if round(seq_length*self.wav_len_per_seq) >= wavs.size(1):
            wavs = zero_padding(wavs, round(seq_length*self.wav_len_per_seq), dim=1)
        wavs = zero_padding(wavs, round(self.wav_len_per_seq)+wavs.size(1), dim=1, t_first=False)

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
        
        

        lstm_out = self.lstm_layer(wav_out)

        tf_input = lstm_out.detach().clone() # transformer can not influence lstm's gradient
        cond_wav_fea = cond_wav_fea.detach()
        tf_out = tf_input + self.transformer_layer(tf_input, cond_wav_fea, seqs_len) # residual connection
        
        
        if self.out_norm is not None:
            if self._norm_check == False or self.out_norm['min'].device != wavs.device:
                dev = wavs.device
                self.out_norm['min'] = self.out_norm['min'].to(dev)
                self.out_norm['max'] = self.out_norm['max'].to(dev)
                self._norm_check = True
            tf_out =  self.out_norm['min'] + (tf_out+1) * 0.5 * (self.out_norm['max'] - self.out_norm['min']) 
            lstm_out = self.out_norm['min'] + (lstm_out+1) *0.5 * (self.out_norm['max'] - self.out_norm['min']) 
        return {'tf_out' : tf_out, 'lstm_out': lstm_out}

