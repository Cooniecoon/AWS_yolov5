from utils.general import xyxy2xywh
import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torchvision
import torchvision.models as models

class DogLSTM(nn.Module):
    def __init__(self,hidden_dim=256,cls_num=5):
        super().__init__()
        torch.manual_seed(1)
        #self.transform = nn.Sequential(torchvision.transforms.Normalize(0.5,0.5))
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(6,hidden_dim,2)

        self.fc1 = nn.Linear(hidden_dim,1024)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer1 = nn.Sequential(self.fc1,nn.ReLU())
        
        self.fc2 = nn.Linear(1024,512)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.layer2 = nn.Sequential(self.fc2,nn.ReLU())
        
        self.layer3 = nn.Linear(512,cls_num)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        

    def forward(self,input):
        #input = self.transform(input)
        #print(input)
        lstm_out, _ = self.lstm(input)
        out = self.layer1(lstm_out[-1])
        out = self.layer2(out)
        out = self.layer3(out)
        return out



class SequeceCollector():
    def __init__(self,seq,min_dt):
        self.seq = []
        self.seq_len = seq
        self.min_time = min_dt

    def get_sequece(self,data):
        if len(data):
            self.miss_cnt = 0
            if len(self.seq):
                if data[-1] - self.seq[-1][-1] > self.min_time:
                    self.seq.append(data)
            else:
                self.seq.append(data)
            if len(self.seq) > self.seq_len:
                self.seq.pop(0)
        else:
            if len(self.seq):
                self.seq.pop(0)
        return self.seq
    
    def show_seq(self):
        for seq in self.seq:
            print(seq)
        print('-'*59)
            
class TargetFinder():
    def __init__(self,miss_limit=30):
        self.miss_limit = miss_limit
        self.miss_cnt = 0
        self.cen_wh = [0.5,0.5,0.1,0.1]
        self.start = False
        self.last = []

    def xyxy_cenwh(self,xyxy):
        return [(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,xyxy[0]-xyxy[2],xyxy[1]-xyxy[3]]
    
    def find(self,bboxes):
        max_err = 1000
        result = []
        for bbox in bboxes:
            cen_wh = self.xyxy_cenwh(bbox)
            err = (abs(self.cen_wh[0] - cen_wh[0]) + abs(self.cen_wh[1] - cen_wh[1]))
            if err < max_err:
                if err > 0.7:
                    if self.start == True:
                        continue
                result = bbox
                max_err = err

        if len(result):
            self.miss_cnt = 0
            self.cen_wh = self.xyxy_cenwh(result)
            self.start = True
        else:
            self.miss_cnt += 1
            if self.miss_cnt > self.miss_limit:
                self.cen_wh = [0.5,0.5,0.1,0.1]
                self.start = False

        return result

class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss
    
    # validation step
    def validation_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))

class DogBreedPretrainedResnet34(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = models.resnet34(pretrained=False)
        self.last_layer = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.last_layer.weight)
        self.resnet.fc = nn.Sequential(self.last_layer,nn.ReLU())
        self.last_fc = nn.Linear(256,5)
    def forward(self, xb):
        out = self.resnet(xb)
        return self.last_fc(out)