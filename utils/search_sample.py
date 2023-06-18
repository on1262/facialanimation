import torch
import os

if __name__ == '__main__':
    output_dir = '/home/chenyutong/facialanimation/Visualize/infer_sample'
    data_list = torch.load('/home/chenyutong/facialanimation/dataset_cache/CREMA-D/cremad_test.pt', map_location='cpu')
    names = set([data['name'] for data in data_list])
    for root,_,files in os.walk(output_dir):
        for file in files:
            if file.endswith('.mp4'):
                k = file.split('.mp4')[0]
                if k in names:
                    print(k, ' test')
                else:
                    print(k,' train')
    print('Done')
    
