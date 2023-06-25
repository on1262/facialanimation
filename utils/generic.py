import numpy as np
import subprocess
import os, importlib, torch

def vertices2nparray(vertex):
    return np.asarray([vertex['x'],vertex['y'],vertex['z']]).T

def multi_imgs_2_video(img_path, audio_path, out_path):
    if audio_path is not None and os.path.exists(audio_path):
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', \
            '-f', 'image2', '-r', '30', '-i', img_path, '-i', audio_path, '-b:v','1M', out_path])
    else:
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2', '-r', '30', '-i', img_path, '-b:v','1M', out_path])

def load_model_dict(model_path, device):
    print('load model from: ', model_path)
    save_dir = os.path.dirname(os.path.dirname(model_path))
    model_name = os.path.split(save_dir)[1]
    Model = importlib.import_module('Model.' + model_name + '.model').Model
    sd = torch.load(model_path)
    return sd, Model