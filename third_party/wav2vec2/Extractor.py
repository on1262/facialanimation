import torch.nn as nn
import torch
from pytorch_revgrad import RevGrad


# TODO: allow >1 batchsize
class ExtractorAVG(nn.Module):
    def __init__(self, emotion_class=6):
        super(ExtractorAVG, self).__init__()
        self.dense = nn.Linear(768, emotion_class)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.2)
        self.activated_layer = 13
        self.avg_mat = nn.Parameter(0.1*torch.ones((self.activated_layer,768)),requires_grad=True)

    def forward(self, hidden:torch.Tensor):
        hidden = hidden.transpose(1,2) #(batch_size, sequence_length, 13, hidden_size).hidden_size=768
        #hidden = hidden[:,:,3:3+self.activated_layer,:]
        hidden = torch.divide(torch.sum(torch.mul(hidden, self.avg_mat), dim=-2),torch.sum(self.avg_mat, dim=0)) # (batch_size, sequence_length, hidden_size)
        x = self.dense(hidden)
        x = x[:,-1,:]
        return x

    def get_m(self):
        return torch.mean(self.avg_mat, dim=1)

class ExtractorLSTM(nn.Module):
    def __init__(self, emotion_class):
        super(ExtractorLSTM, self).__init__()
        self.activated_layer = 13
        self.avg_mat = nn.Parameter(0.1*torch.ones((self.activated_layer,768)),requires_grad=True)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.2)
        self.lstm = nn.LSTM((768), (128), proj_size=emotion_class, batch_first=True)
        

    def forward(self, hidden:torch.Tensor):
        # feature
        hidden = hidden.transpose(1,2) #(batch_size, sequence_length, 13, hidden_size).hidden_size=768
        #hidden = hidden[:,:,3:3+self.activated_layer,:]
        hidden = torch.divide(torch.sum(torch.mul(hidden, self.avg_mat), dim=-2),torch.sum(self.avg_mat, dim=0)) 
        # hidden: (batch_size, sequence_length, hidden_size)
        # label predictor
        x = self.dp(hidden)
        x,_ = self.lstm(x) #(N,L,Hidden_size)
        x = x[:,-1,:]
        return {'label':x}

    def get_m(self):
        return torch.mean(self.avg_mat, dim=1)

class ExtractorRevGrad(nn.Module):
    def __init__(self, emotion_class, dataset_label=3, alpha=1):
        super(ExtractorRevGrad, self).__init__()
        self.activated_layer = 13
        self.dataset_label = dataset_label
        self.avg_mat = nn.Parameter(0.1*torch.ones((self.activated_layer,768)),requires_grad=True)
        self.domain_dense1 = nn.Linear(768, 128)
        self.domain_dense2 = nn.Linear(128, self.dataset_label)
        self.revgrad = RevGrad(alpha=alpha)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.2)
        self.lstm = nn.LSTM((768), (128), proj_size=emotion_class, batch_first=True)
        

    def forward(self, hidden:torch.Tensor):
        # feature
        hidden = hidden.transpose(1,2) #(batch_size, sequence_length, 13, hidden_size).hidden_size=768
        #hidden = hidden[:,:,3:3+self.activated_layer,:]
        hidden = torch.divide(torch.sum(torch.mul(hidden, self.avg_mat), dim=-2),torch.sum(self.avg_mat, dim=0)) # (batch_size, sequence_length, hidden_size)
        # domain classifier
        d = self.revgrad(hidden)
        d = self.domain_dense1(d)
        d = self.dp(d)
        d = self.relu(d)
        d = self.domain_dense2(d)
        d = torch.mean(d, dim=1)
        # label predictor
        x = self.dp(hidden)
        x,_ = self.lstm(x) #(N,L,Hidden_size)
        x = x[:,-1,:]
        return {'label':x, 'domain':d}
    
    def get_m(self):
        return torch.mean(self.avg_mat, dim=1)
