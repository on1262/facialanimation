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
predict whole sequence
version2: n_layer=4
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

class ScaledRelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def square_mat(self, length_max):
        range_vec = torch.arange(length_max)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        return torch.LongTensor(distance_mat_clipped + self.max_relative_position)

    def forward(self, length_q, length_k):
        square_mat = self.square_mat(max(length_q, length_k))
        if length_k == length_q:
            final_mat = square_mat
        elif length_k > length_q:
            coeff = length_k / length_q
            final_mat = torch.zeros((length_q, length_k), dtype=torch.long)
            for idx in range(final_mat.size(0)):
                final_mat[idx,:] = square_mat[math.floor(idx*coeff),:]
        elif length_k < length_q:
            coeff = length_q / length_k
            final_mat = torch.zeros((length_q, length_k), dtype=torch.long)
            for idx in range(final_mat.size(1)):
                final_mat[:,idx] = square_mat[:,math.floor(idx*coeff)]
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, max_relative_position=16):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        self.relative_position_k = ScaledRelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = ScaledRelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k).to(query.device)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / (self.scale.to(query.device))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v).to(query.device)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        #x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        return x


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
        self.attention = MultiHeadAttentionLayer(n_heads=attn_heads, hid_dim=hidden, dropout=dropout, max_relative_position=16)
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
        self.out_linear = nn.Linear(hidden, n_params_fea)
        self.out_activ = nn.Tanh()
        self.wav_linear = nn.Linear(n_wav_fea, hidden)

        # multi-layers transformer blocks, deep network
        self.cross_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, mode='cross') for _ in range(n_layers)])
        
        self.self_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, mode='self') for _ in range(n_layers)])

    def forward(self, x, c):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        c = self.wav_linear(c)
        x = self.input_linear(x)

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
            n_params_fea=params_channels, n_wav_fea=self.wav_fea_channels, hidden=768, n_layers=4, attn_heads=6, dropout=dp)
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
        tf_out = tf_input + self.transformer_layer(tf_input, cond_wav_fea) # residual connection
        
        
        if self.out_norm is not None:
            if self._norm_check == False or self.out_norm['min'].device != wavs.device:
                dev = wavs.device
                self.out_norm['min'] = self.out_norm['min'].to(dev)
                self.out_norm['max'] = self.out_norm['max'].to(dev)
                self._norm_check = True
            tf_out =  self.out_norm['min'] + (tf_out+1) * 0.5 * (self.out_norm['max'] - self.out_norm['min']) 
            lstm_out = self.out_norm['min'] + (lstm_out+1) *0.5 * (self.out_norm['max'] - self.out_norm['min']) 
        return {'tf_out' : tf_out, 'lstm_out': lstm_out}

