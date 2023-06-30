from dataset import VOCASET, EnsembleDataset, BaselineVOCADataset, LRS2Dataset
from utils.interface import EMOCAModel, DANModel
import torch

def dataset_cache():
    vocaset_train = VOCASET('train')
    vocaset_valid = VOCASET('valid')
    vocaset_test = VOCASET('test')

    device = torch.device('cuda:2')
    emoca = EMOCAModel(device=device)
    dan = DANModel(device=device)
    ens_train = EnsembleDataset('train', device=device, emoca=emoca, dan=dan)
    # ens_valid = EnsembleDataset('valid')