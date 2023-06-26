import numpy as np
import subprocess
import os, importlib, torch
import glob

def vertices2nparray(vertex):
    return np.asarray([vertex['x'],vertex['y'],vertex['z']]).T

def multi_imgs_2_video(img_path, audio_path, out_path):
    if audio_path is not None and os.path.exists(audio_path):
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', \
            '-f', 'image2', '-r', '30', '-i', img_path, '-i', audio_path, '-b:v','1M', out_path])
    else:
        subprocess.run(['ffmpeg' , '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2', '-r', '30', '-i', img_path, '-b:v','1M', out_path])

def load_model_dict(model_path, device):
    '''load model from model dir, e.g. ./models/tf_emo_4/'''
    print('load model from: ', model_path)
    save_dir = os.path.dirname(model_path)
    model_name = os.path.split(save_dir)[1]
    Model = importlib.import_module('models.' + model_name + '.model').Model
    latest_file = latest_file_path(os.path.join(model_path, 'saved_model'))
    sd = torch.load(latest_file)
    return sd, Model


def latest_file_path(dir_path, suffix='.pth'):
    '''get latest file path with specific suffix'''
    list_of_files = glob.glob(os.path.join(dir_path, '*'+suffix)) # * means all if need specific format then *.csv
    latest_path = max(list_of_files, key=os.path.getctime)
    return latest_path