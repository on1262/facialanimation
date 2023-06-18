import torch
import time

class MemCheck():
    def __init__(self, debug, dev):
        self.dev = dev
        self.debug = debug
        self.stage_list = []
        self.max_usage = []
        self.cur_usage = []
        self.time_consume = []
    
    def log(self, stage:str):
        if self.debug > 0:
            self.stage_list.append(stage)
            self.max_usage.append(torch.cuda.max_memory_allocated(self.dev) / (1024*1024*1024))
            self.cur_usage.append(torch.cuda.memory_allocated(self.dev) / (1024*1024*1024))
            self.time_consume.append(time.time())

    def clear(self):
        self.stage_list.clear()
        self.max_usage.clear()
        self.cur_usage.clear()
        self.time_consume.clear()
    
    def summary(self):
        if self.debug > 0:
            print('='*10, 'mem summary: ', self.dev, '='*10)
            for idx,s in enumerate(self.stage_list):
                print('stage ', s, '\t current: ', self.cur_usage[idx], ' GB \t max: ', \
                    self.max_usage[idx], ' GB', ' \t time stamp=', self.time_consume[idx] - self.time_consume[0])
            print('='*30)


