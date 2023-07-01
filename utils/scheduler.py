import torch.nn as nn


class PlateauDecreaseScheduler():
    '''
    no warm up mode(set warm up steps=0):
        initial learning rate: init_lr*lr_coeff
        learning rate linearly get closed to min_lr when steps increasing.
    
    warm up mode:
        init_lr is not used
        learning rate goes to warmup_lr when step=warmmp_steps, then mixed with min_lr.
          

    '''
    def __init__(self, 
    optimizers_list:list, 
    lr_coeff_list=None, 
    warmup_steps=100, 
    warmup_lr=1e-3, 
    warmup_enable_list=[True],
    factor=0.2,
    init_lr=1e-4,
    min_lr=1e-5,
    patience=3
    ):
        self.optims = optimizers_list
        self.patience = patience
        if lr_coeff_list is not None:
            self.lr_coeff_list = lr_coeff_list
        else:
            self.lr_coeff_list = [1 for _ in optimizers_list]
        
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.factor = factor

        assert(len(optimizers_list) == len(warmup_enable_list))
        self.warm_up_config = {
            'enable_list' : warmup_enable_list, 'init_lr':init_lr, 'warm_up_steps': warmup_steps, 'warm_up_lr': warmup_lr
            }
        self.now_warm_up_steps = 0
        self.lr_list = [c*init_lr for c in self.lr_coeff_list]
        for idx, opt in enumerate(self.optims):
            for g in opt.param_groups:
                g['lr'] = self.lr_list[idx]
        
    def log_loss(self, tr_loss, te_loss, epoch):
        if self.loss_dict['best_test_loss'] is None: # all none
            self.loss_dict['best_test_loss'] = te_loss
            self.loss_dict['best_train_loss'] = tr_loss
            self.loss_dict['best_test_epoch'] = epoch
            self.loss_dict['best_train_epoch'] = epoch
        else:
            if tr_loss < self.loss_dict['best_train_loss']:
                self.loss_dict['best_train_loss'] = tr_loss
                self.loss_dict['best_train_epoch'] = epoch
                self.lr_decreased = False
            if te_loss < self.loss_dict['best_test_loss']:
                self.loss_dict['best_test_loss'] = tr_loss
                self.loss_dict['best_test_epoch'] = epoch
                self.lr_decreased = False

    def get_lr(self):
        return self.lr_list

    def get_wmp_step(self):
        return self.now_warm_up_steps

    def step(self, epoch):
        if self.now_warm_up_steps < self.warm_up_config['warm_up_steps']:
            lr = self.warm_up_config['warm_up_lr'] * (self.now_warm_up_steps / self.warm_up_config['warm_up_steps'])
        else:
            all_step = self.now_warm_up_steps + self.warm_up_config['warm_up_steps']
            lr = (self.warm_up_config['warm_up_steps'] / all_step) * self.warm_up_config['warm_up_lr'] + (self.now_warm_up_steps / all_step) * self.min_lr
        self.now_warm_up_steps += 1
        for opt in self.optims:
            for g in opt.param_groups:
                g['lr'] = lr

        self.lr_list = [lr for _ in range(len(self.lr_list))]


