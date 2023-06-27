import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils.converter import video2sequence, convert_img, save_img
import os.path as path
import argparse

def plot_curve(emo_tensor, labels, save_path):
    emo_tensor = emo_tensor.to('cpu')
    # plot
    fig, ax = plt.subplots(figsize=(15,5), dpi=160, sharex=True, sharey=True)
    x = range(emo_tensor.size(0))
    for idx in range(emo_tensor.size(1)):
        ax.plot(x, emo_tensor[:, idx], label=labels[idx])
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)

def check_curve(dan_model, video_path, output_folder='.'):
    img_path_list = video2sequence(video_path)
    result_list= []
    imgs = None
    for idx,p in enumerate(img_path_list):
        out = dan_model.inference(convert_img(p, 'tensor'))
        result_list.append(out.detach().cpu()) # size=1,7
    result = torch.cat(result_list, dim=0) # seq_len,7
    print('result', result.size())
    plot_curve(result, dan_model.labels, path.join(output_folder, path.split(video_path)[1].split('.')[0] + '_curve.png'))
    print('.', end='')


if __name__ == '__main__':
    from interface import DANModel
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--output_folder', type=str, default='.')
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.device))
    if args.output_folder != '.':
        os.makedirs(args.output_folder, exist_ok=True)

     # load DAN
    dan_model = DANModel(device=device)

    # find video
    for root, dirs, files in os.walk(args.video_path):
        for name in files:
            if name.endswith('.flv'):
                check_curve(dan_model, os.path.join(args.video_path, name), args.output_folder)
    print('Done')
        




    
