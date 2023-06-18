import librosa
import numpy
import torch
import os
import argparse
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate .wav file into preload .pt file')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    train_dataset_path = '/home/chenyutong/facialanimation/dataset_cache/CREMA-D'
    debug_dataset_path = '/home/chenyutong/facialanimation/DECA/TestSamples/cremad_test'
    dataset_path = debug_dataset_path if args.debug else train_dataset_path
    audio_dict = {}
    for root, dirs, files in os.walk(join(dataset_path, 'AudioWAV')):
        for idx, name in enumerate(files):
            print(idx, ': ' , name)
            if name.endswith('.wav'):
                data, _ = librosa.load(join(root, name),sr=16000)
                wav_tensor = torch.from_numpy(data.T)
                audio_dict[name.split('.')[0]] = wav_tensor.squeeze()
    torch.save(audio_dict, join(dataset_path, 'wav.pt'))
    print('done')
