import os
from os.path import join as osjoin
import yaml

class Configs():
    def __init__(self, config_dir) -> None:
        self.conf = {}
        self.config_dir = config_dir
        # load yaml from config dir
        ymls = ['global', 'inference', 'dataset', 'trainer']
        for name in ymls:
            with open(osjoin(config_dir, name + '.yml')) as f:
                self.conf[name] = yaml.load(f, Loader=yaml.SafeLoader)
        # load yaml from model dir
        model_path = self.conf['global']['paths']['model']
        for model_name in os.listdir(model_path):
            model_conf = os.path.join(model_path, model_name, 'config.yml')
            if os.path.exists(model_conf):
                with open(model_conf) as f:
                    self.conf['model'][model_name] = yaml.load(f, Loader=yaml.SafeLoader)
        
    
    def __getitem__(self, name):
        return self.conf[name].copy()
    

GBL_CONF = Configs('.', '../configs')
PATH = GBL_CONF['global']['paths']