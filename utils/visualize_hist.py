import sys
sys.path.append('/home/chenyutong/facialanimation')
from utils.interface import EMOCAModel, DANModel
from dataset import EnsembleDataset
import torch

if __name__ == '__main__':
    device = torch.device('cuda:7')
    label_dict={'NEU':0,'HAP':1,'ANG':2,'SAD':3,'DIS':4,'FEA':5,'EXC':1}
    emoca = EMOCAModel(device=device, decoder_only=False)
    dan = DANModel(device=device)
    self.dataset_train = EnsembleDataset(
        self.dataset_path, label_dict, return_domain=False, dataset_type='train', device=device, emoca=emoca, debug=0)
