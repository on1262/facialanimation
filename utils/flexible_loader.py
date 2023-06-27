import torch
import math


class FlexibleLoader():
    '''Dataloader with flexible batch size'''
    def __init__(self, dataset, batch_size:int, sampler, collate_fn, clip_max=True):
        self.dataset = dataset
        self.batch_size = batch_size
        if sampler is not None:
            self._sampler = sampler
        else:
            self._sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self._it_sampler = None
        self._collate_fn = collate_fn
        self.clip_max = clip_max
        self._count = 0

    def __iter__(self):
        self._count = 0
        self._it_sampler = iter(self._sampler)
        return self
    
    def __next__(self):
        if self._count >= len(self._sampler):
            raise StopIteration()
        now_list = []
        assert(self.batch_size > 0)
        try:
            for _ in range(self.batch_size):
                now_list.append(next(self._it_sampler))
        except StopIteration:
            self._count = len(self._sampler)
        
        self._count += len(now_list)

        # batch_size
        return self._collate_fn([self.dataset[idx] for idx in now_list], self.clip_max)
    
    # not reliable result
    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

def test_collate_fn(data_list:list):
    return data_list

if __name__ == '__main__':
    import torch.utils.data.sampler as sampler
    a = [1,2,3,4,5,6,7,8,9,10]
    s = sampler.SequentialSampler(a)
    loader = FlexibleLoader(a, batch_size=2, sampler=s, collate_fn=test_collate_fn)
    for data in loader:
        loader.batch_size += 1
        print(data)
    
    
