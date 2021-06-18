import torch
import torch.nn as nn

class DogLSTM(nn.Module):
    def __init__(self,hidden_dim=256,cls_num=5):
        super().__init__()
        torch.manual_seed(1)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(5,hidden_dim,2)
        self.layer1 = nn.Linear(hidden_dim,128)
        self.layer2 = nn.Linear(128,cls_num)

    def forward(self,input):
        lstm_out, _ = self.lstm(input)
        out = self.layer1(lstm_out[-1])
        out = self.layer2(out)
        return out