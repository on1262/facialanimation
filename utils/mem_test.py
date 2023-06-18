import torch
import torch.nn.functional as F

class test():
    def __init__(self):
        self.a = {'new_key': torch.randn(1000,1000).to('cuda:7'), 'others': torch.randn(4,4)}

    def get_item(self):
        result_dict = self.a
        p = torch.randn(5,5).to('cuda:7')
        p = p + p
        result_dict['new_key'] = p
        return result_dict

if __name__ == '__main__':
    init = torch.randn(5,5).to('cuda:7')
    ca = test()
    start_mem = torch.cuda.memory_allocated('cuda:7')
    for _ in range(100):
        d = ca.a
        d['new_key'] = torch.randn(1000,1000).to('cuda:7')

    end_mem = torch.cuda.memory_allocated('cuda:7')
    print('delta=', end_mem-start_mem)
    
