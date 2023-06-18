import torch

def calculate_norm(dataset):
    out_norm = None
    for idx in range(len(dataset)):
        params = dataset[idx]['params']
        if out_norm is None:
            out_norm = {'max' : torch.max(params, dim=0).values, 'min' : torch.min(params, dim=0).values}
        else:
            out_norm['max'] = torch.maximum(out_norm['max'], torch.max(params, dim=0).values)
            out_norm['min'] = torch.minimum(out_norm['min'], torch.min(params, dim=0).values)
        if idx % 50 == 0:
            print('.', end='')
    #print('max: ', out_norm['max'].max(dim=0).values, 'min: ', out_norm['min'].min(dim=0).values)
    for idx in range(out_norm['max'].size(0)):
        print('idx=', idx, ' \t max=', out_norm['max'][idx], ' \t min=', out_norm['min'][idx])
    return out_norm

if __name__ == '__main__':
    data_list = torch.load('/home/chenyutong/facialanimation/dataset_cache/CREMA-D/cremad_train.pt', map_location='cpu')

    norm = calculate_norm(data_list)
    print('norm')
    torch.save(norm, './new_norm.pt')
    











