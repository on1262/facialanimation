import torch
import numpy as np
import os
from utils.config_loader import PATH
from matplotlib import pyplot as plt

def hist_inv_func(hist:torch.Tensor, data_min, data_max, X:torch.Tensor, max_amp=10):
    hist = hist.view(-1)
    X = X.view(-1)
    X = torch.floor(hist.size(0) * (X - data_min) / (data_max-data_min))
    # TODO X out of bound should be 0
    X = torch.clip(X, 0, hist.size(0)-1).long()
    return torch.clip(1 / hist[X], min=0, max=max_amp)

def hist_inv_func_dict(hist_dict:dict, X:torch.Tensor, max_amp=10):
    return hist_inv_func(hist_dict['hist'], hist_dict['min'], hist_dict['max'], X, max_amp)


def cal_hist(data:torch.Tensor, bins=100, save_fig=False, name_list=None):
    data = data.to('cpu')
    if data.dim() == 2:
        return [cal_hist(data[idx,:], bins=bins, save_fig=save_fig, name_list=(None if name_list is None else [name_list[idx]])) for idx in range(data.size(0))]
    assert(data.dim() == 1)
    data_min, data_max = data.min().item(), data.max().item()
    # cal hist
    hist = torch.histc(data, bins=bins, min=data_min, max=data_max) # (num of data)
    hist[hist < 1] = 1 # avoid div 0
    hist = hist / hist.mean()
    if save_fig:
        fig, ax = plt.subplots(figsize=(10,5), dpi=160)
        # plot hist
        x = (data_min + (np.arange(0.5, bins+0.5) / bins) * (data_max - data_min))
        y = hist.numpy()
        ax.bar(x, y, width=(data_max-data_min)/bins, edgecolor="white", linewidth=(data_max-data_min)*2/bins, alpha=0.5)
        ax.set(xlim=(data_min, data_max), xticks=((np.arange(0,bins,bins/10) + 0.5) / bins) * (data_max - data_min) + data_min,
               ylim=(0, y.max()), yticks=np.arange(0, y.max(), y.max() / 10))
        # plot regression result
        y_func = hist_inv_func(hist, data_min, data_max, torch.from_numpy(x))
        ax.plot(x, y_func.numpy(), linewidth=2.0)
        os.makedirs(os.path.join(PATH['inference']['visualize'], 'hist'), exist_ok=True)
        file_name = 'hist' if name_list is None else ('hist_' + name_list[0])
        plt.savefig(os.path.join(PATH['inference']['visualize'], 'hist', file_name + '.png'))
        print('saved as' , os.path.join(PATH['inference']['visualize'], 'hist', file_name + '.png'))
        plt.close(fig)
    return {'hist':hist,'min':data_min,'max':data_max}

if __name__ == '__main__':
    count = 3000
    cal_hist(torch.randn(2, count), bins=30, save_fig=True, name_list=['a','b'])
