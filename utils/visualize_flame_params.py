import torch
from interface import EMOCAModel
from converter import save_img, convert_img
import os

if __name__ == '__main__':
    counts = 200
    out_folder = os.path.join('.', 'visualize_params')
    os.makedirs(out_folder, exist_ok=True)
    emoca = EMOCAModel(device='cuda:7', decoder_only=True)
    code_dict = {'expcode':torch.zeros(counts,50),'shapecode': torch.zeros(counts,100), 'posecode':torch.zeros(counts,6), 'cam': torch.zeros(counts,3)}
    for idx in range(counts):
        if idx >= 100:
            code_dict['shapecode'][idx,idx-100] = -3
        else:
            code_dict['shapecode'][idx, idx] = 3
        code_dict['cam'][idx,0] = 10
    imgs = emoca.decode(code_dict, {'geo'}, target_device='cpu')['geometry_coarse']
    for idx in range(counts):
        name = -(idx-counts // 2) if idx > counts // 2 else idx
        if idx == counts // 2:
            name = '-0'
        save_img(convert_img(imgs[idx,...], 'emoca', 'tvsave'), os.path.join(out_folder,f'{name}.png'))
    print('Done')
    # expression params idx related to mouth open:
    # [1, 3, 5]
    # expression params idx related to mourh horizontal movement:
    # [0, 1, 3, 6, 7]
    # only first 20 expression params is useful for expression generation

