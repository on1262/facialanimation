import torch.nn as nn
from GST.feature_extract import LogMelFbank, GlobalMVN
from GST.style_encoder import StyleEncoder
import torch

class gst_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats_extract = LogMelFbank(
            fs = 24000,
            n_fft = 2048,
            win_length = 1200,
            hop_length = 300,
            window = "hann",
            center = True,
            normalized = False,
            onesided = True,
            n_mels = 80,
            fmin = 80,
            fmax = 7600,
            htk = False,
            log_base = 10.0)
        
        self.normalize = GlobalMVN('/home/chenyutong/facialanimation/GST/feats_stats.npz')
        
        self.gst = StyleEncoder(
            idim=80,  # the input is mel-spectrogram
            gst_tokens=128,
            gst_token_dim=512,
            gst_heads=8)

        
        
    def forward(self, speech, speech_lengths):
        feats, feats_lengths = self.feats_extract(speech, speech_lengths)
        feats, feats_lengths = self.normalize(feats, feats_lengths)
        feats = feats[:, : feats_lengths.max()]
        style_ebd = self.gst(feats)
        return style_ebd


if __name__ == '__main__':
    print('loading model')
    model = GSTModel().to('cpu')
    model.load_state_dict(torch.load('gst_state_dict.pth'), strict=False)
    #torch.save(model.state_dict(), 'gst_state_dict.pth')
    print('testing')
    out = model(torch.randn(3,22400), torch.LongTensor(data=[22400,22400,22400]))
    print(out.size())

