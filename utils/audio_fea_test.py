import torch
from converter import audio2tensor
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    wav = audio2tensor('test/stft_test.wav', 16000)
    fps = 30
    bin_per_frame = 3
    freq_bins = 100
    n_fft = round(freq_bins-1) * 2 # out is (200 / 2) + 1
    hop = round(16000/(bin_per_frame*fps)) # points per bin in x axis of output, 
    win_len = n_fft

    out = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win_len, window=torch.hann_window(win_len), return_complex=True)
    out_abs = torch.abs(out)
    print('wav size:', wav.size(), ' out size:', out_abs.size())

    fig, ax = plt.subplots(figsize=(15,5), dpi=160)
    # bins actually returns middle value of each chunk
    # so need to add an extra element at zero, and then add first to all
    plt.pcolormesh(np.log10(out_abs+1e-6), cmap='plasma', vmin=-1.5, vmax=np.log10(out_abs+1e-6).max())
    plt.title("pcolormesh post adj.")
    plt.savefig('spec.png')
    
