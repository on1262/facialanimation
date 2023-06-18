import librosa
import cv2
import torch
import os
import subprocess
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from torchvision.utils import save_image

"""
image format converter
format:
    channel code: 3hw or hw3 (auto detect)
    color code: RGB or BGR
    range code: 01 or -11 or 255 (auto detect)
    type code: t(tensor) or n(ndarray)
    type code 2: u(uint8) or f(float)
    shortcut: see shortcuts 
    keep_batch: keep dim = 4 even if batch = 1

"""
def convert_img(i_img, i_code='RGB', o_code='3hwRGB01_tf', keep_batch=False):
    shortcuts = {'pil':'3hwRGB01_tf',
        'sk':'hw3RGB255_nu', 
        'cv2':'hw3BGR255_nu', 
        'fan_in':'hw3BGR255_tu',
        'fan_out':'3hwBGR01_tf',
        'tv':'3hwRGB255_tu',
        'tvsave':'3hwRGB01_tf',
        'emoca':'3hwRGB01_tf',
        'dan':'3hwRGB01_tf', 
        'mtcnn':'RGB255_tu',
        'store':'3hwRGB255_tu'
        }
    # step 1: check input
    i_code = shortcuts[i_code] if i_code in shortcuts else i_code
    o_code = shortcuts[o_code] if o_code in shortcuts else o_code
    if o_code in i_code and not isinstance(i_img, Image.Image):
        return i_img
    
    # step 2: convert to tensor
    if isinstance(i_img, np.ndarray):
        i_img = torch.from_numpy(i_img)
    elif isinstance(i_img, Image.Image):
        i_img = F.to_tensor(i_img)
    else:
        if not isinstance(i_img, torch.Tensor):
            print('convert_img: unsupported input image type: ', type(i_img))
            return None
    
    i_img = i_img.detach().clone()
    # step 3: convert input image to batch*3hwRGB01_tf
    if i_img.dim() == 3:
        i_img = i_img.unsqueeze(0)
    if i_img.size(-1) == 3 and ('hw3' in i_code or i_img.size(1) != 3):
        i_img = i_img.permute(0,3,1,2)
    if 'BGR' in i_code:
        i_img = i_img[:,[2,1,0],...]
    if i_img.dtype == torch.uint8:
        i_img = i_img.float()
    if '255' in i_code or i_img.max() > 1.5:
        i_img = i_img / 255.0
    assert(i_img.min() > -1e-6)
    
    # step 4: convert input image to output format
    #o_img = i_img.clone()
    o_img = i_img
    if '255' in o_code:
        o_img = torch.clip(o_img * 255, 0, 255)
    elif '01' in o_code:
        o_img = torch.clip(o_img, 0, 1)
    if 'u' in o_code:
        o_img = o_img.type(torch.uint8) # period + floor. e.g. 19->19, 256->0, -1->255, 1.9->1
    if 'BGR' in o_code:
        o_img = o_img[:, [2,1,0],...]
    if 'hw3' in o_code:
        o_img = o_img.permute(0,2,3,1)
    if keep_batch == False and o_img.size(0) == 1:
        o_img = o_img.squeeze(0)

    o_img = o_img.contiguous()
    if 'n' in o_code:
        o_img = o_img.numpy()
    
    return o_img

# format: tv
def save_img(input, save_path):
    save_image(input, save_path)

    
def video2sequence(video_path, sample_step=1, return_path=True, o_code='emoca') -> list:
    dir_name, file_name = os.path.split(video_path)
    if return_path:
        videofolder = os.path.join(dir_name, 'frames', file_name.split('.')[0])
        os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    result_list = []
    while success:
        if count%sample_step == 0:
            if return_path:
                imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
                cv2.imwrite(imagepath, image)     # height, weight, 3(channel=BGR)
                result_list.append(imagepath)
            else:
                img = convert_img(image, i_code='cv2', o_code=o_code, keep_batch=False)
                result_list.append(img)
        count += 1
        success, image = vidcap.read()
    if return_path:
        print(f'video frames are stored in {videofolder}')
        return sorted(result_list)
    else:
        return result_list

def audio2tensor(audio_path:str, sr=16000):
    p = None
    if os.path.isfile(audio_path):
        p = audio_path
    elif os.path.isdir(audio_path):
        for root,_,files in os.walk(audio_path):
            for name in files:
                if name.endswith('wav'):
                    p = os.path.join(root,name)
                    print('load audio:', p)
                    break
    else:
        print('audio to tensor: path should be dir or file, but get ', audio_path)
        return None
    data, sr = librosa.load(p, mono=True, sr=16000)
    wav_tensor = torch.from_numpy(data.T).squeeze()
    return wav_tensor

def video2wav(vid_path:str, cache_dir_path:str): 
    _, name = os.path.split(vid_path)
    name = name.split('.')[0] + '.wav'
    out_path = os.path.join(cache_dir_path, name)
    subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y','-i', vid_path, '-ar', '16000', out_path])
    return out_path

