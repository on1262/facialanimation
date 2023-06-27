import torch, os
from tqdm import tqdm
from utils.config_loader import PATH 

def calculate_norm(dataset):
    out_norm = None
    for idx in tqdm(range(len(dataset))):
        params = dataset[idx]['params']
        if out_norm is None:
            out_norm = {'max' : torch.max(params, dim=0).values, 'min' : torch.min(params, dim=0).values}
        else:
            out_norm['max'] = torch.maximum(out_norm['max'], torch.max(params, dim=0).values)
            out_norm['min'] = torch.minimum(out_norm['min'], torch.min(params, dim=0).values)
    for idx in range(out_norm['max'].size(0)):
        print('idx=', idx, ' \t max=', out_norm['max'][idx], ' \t min=', out_norm['min'][idx])
    return out_norm

if __name__ == '__main__':
    path = os.path.join(PATH['dataset']['cache'], 'cremad', 'cremad_train.pt')
    data_list = torch.load(path, map_location='cpu')

    norm = calculate_norm(data_list)
    print('norm')
    torch.save(norm, os.path.join(PATH['dataset']['cache'], 'new_norm.pt'))
    











