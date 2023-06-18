import torch
from decimal import Decimal
from interface import EMOCAModel
from converter import save_img, convert_img
import os
import json

'''
upper lip: 3547
left corner: 2845, right corner: 1730
bottom lip: 3513
'''
def dump_flame():
    emoca = EMOCAModel(device='cuda:4', decoder_only=True)
    code_dict = {'expcode':torch.zeros(1,50),'shapecode': torch.zeros(1,100), 'posecode':torch.zeros(1,6), 'cam': torch.zeros(1,3)}
    code_dict['cam'][0,0] = 10
    code_dict['posecode'][0,3] = 0.2
    outs = emoca.decode(code_dict, return_verts=True, return_faces=True, target_device='cpu')
    outs['verts'] = outs['verts'].squeeze(0)
    print(outs['verts'].size(0), ' size0')
    for idx in range(outs['verts'].size(0)):
        outs['verts'][idx] = outs['verts'][idx] * 100 # scale 100x

    with open('face.log', mode='w') as f:
        for idx in range(outs['verts'].size(0)):
            print(idx, ' ', outs['verts'][idx], file=f)

    print('verts', outs['verts'].size(), ' faces:', outs['faces'].size())
    with open('face.json', mode='w') as f:
        json.dump({'verts':outs['verts'].tolist(), 'faces':outs['faces'].tolist()}, f)
    print('done')

def dump_biwi():
    with open('templete/BIWI.vl', mode='rb') as f:
        count = f.read(4) # unsigned long
        for i in range(count):
            

if __name__ == '__main__':
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
