#!/usr/bin/env python
import torch
import numpy as np
from numpy import random
import socket
from PIL import Image
from torch._C import device
from torchvision import transforms

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box

import cv2
import sys
import time
import copy
from collections import defaultdict

from socket_funcs import *
from models.DogLSTM import *


def image_padding(img,dsize):
    ht, wd, cc= img.shape

    ww = dsize[0]
    hh = dsize[1]
    if wd<=ww and ht<=hh:
        color = (0,0,0)
        result = np.full((hh,ww,cc), color, dtype=np.uint8)

        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        result[yy:yy+ht, xx:xx+wd] = img
        
        # result[:,:xx]=cv2.flip(img[:,:xx], 1)
        # result[:,xx+wd:]=cv2.flip(img[:,(wd-xx):], 1)

        # result[:yy,:]=cv2.flip(img[:yy,:], 1)
        # result[yy+ht:,:]=cv2.flip(img[(ht-yy):,:], 1)

    elif wd>ww:
        img=cv2.resize(img,(0,0),fx=ww/wd,fy=ww/wd,interpolation=cv2.INTER_LINEAR)
        result=image_padding(img,dsize=(ww,hh))

    elif ht>hh:
        img=cv2.resize(img,(0,0),fx=hh/ht,fy=hh/ht,interpolation=cv2.INTER_LINEAR)
        result=image_padding(img,dsize=(ww,hh))

    return result


def preprocessing(img):
    # img = image_padding(img)
    img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to("cuda:0")
    img = img.half()
    img /= 255.0 
    img = img.unsqueeze(0)
    return img

def rename(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)

def predict_breed(img,model):
    # img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    img = image_padding(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    trans=transforms.ToTensor()
    img=trans(img).to(dev)
    # img /= 255.0 
    img = img.unsqueeze(0)
    pred = model(img)[0]
    _,predicted = torch.max(pred,dim=0)
    predicted=int(predicted.cpu())
    print(predicted)
    

    '''
    Chihuahua 17
    
    Pomeranian 53

    Cardigan 15
    Pembroke 52

    Labrador_retriever 40
    curly-coated_retriever 90
    flat-coated_retriever 93
    golden_retriever 95
    '''
    
    # 'Welsh_corgi'
    if predicted == 52 or predicted == 17:
        return 2
    # 'Chihuahua',
    elif predicted == 17:
        return 0
    # 'Pomeranian'
    elif predicted == 53:
        return 1
    # 'golden_retriever'
    elif predicted == 40 or predicted == 90 or predicted == 93 or predicted == 95:
        return 4
    # 'etc'
    else:
        return 3


with open('AWS_IP.txt', 'r') as f:
    TCP_IP = f.readline()
TCP_PORT = 6666
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
    
TCP_PORT = 5555
ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind((TCP_IP, TCP_PORT))
ss.listen(True)

print('listening...')
cam_client, addr = s.accept()
print('image node connected')
msg_client, addr = ss.accept()
print('message node connected')
print("start")

dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'etc', 'golden_retriever']
# dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'golden_retriever']

dog_size={'golden_retriever' : 'big', 'Welsh_corgi' : 'middle', 'Chihuahua' : 'small', 'Pomeranian' : 'small', 'etc' : 'None'}


if __name__ == "__main__":
    # YOLO
    model_path = "./weights/06_20.pt"
    print('Load YOLO model')
    model = attempt_load(model_path, map_location="cuda")
    model = model.autoshape()
    model.half()
    names = model.module.names if hasattr(model, "module") else model.names
    print("classes : ",names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Behavior LSTM
    dev = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('Load LSTM model')
    behavior = DogLSTM().to(dev)
    behavior.load_state_dict(torch.load('./weights/DogLSTM.pt', map_location=dev))
    behavior.eval()
    seqCollector = SequeceCollector(seq=15,min_dt=100)

    # Breed Classifier
    weights_fname = './weights/breedClassifier_120cls_res34.pt'
    
    print('Load Classifier model')
    breed_clf = DogBreedPretrainedResnet34()
    
    breed_clf.to(dev)
    breed_clf.load_state_dict(torch.load(weights_fname))
    breed_clf.eval()

    breeds = []
    for n in dog_breeds:
        breeds.append(rename(n))

    targetFinder = TargetFinder()

    img_roi = np.array([[[255,255,255]]], dtype=np.uint8)
    breed=0

    start_time = time.time()
    with torch.no_grad():
        while True:
            t = time.time()
            im0 = recv_img_from(cam_client)
            h,w=im0.shape[:2]
            img = preprocessing(im0)

            # Inference
            prediction = model(img)[0]
            prediction = non_max_suppression(prediction)
            prediction = prediction[0].cpu().numpy()
            bboxes = []
            for pred in prediction:
                if pred is not None:
                    x1 = min(1,max(0,float(pred[0]/640)))
                    y1 = min(1,max(0,float(pred[1]/640)))
                    x2 = min(1,max(0,float(pred[2]/640)))
                    y2 = min(1,max(0,float(pred[3]/640)))
                    cls = int(pred[-1])
                    bbox=[x1, y1, x2, y2, cls,(time.time() - start_time)*1000]
                    bboxes.append(bbox)

            target = targetFinder.find(bboxes)
            seq_target = copy.deepcopy(target)

            if len(target)>10:
                if target[4] != 0:
                    seq_target[0] *= 640
                    seq_target[1] *= 480
                    seq_target[2] *= 640
                    seq_target[3] *= 480
                    seq_target[4] -= 1 
                    seq = seqCollector.get_sequece(seq_target)
                
                    if len(seq):
                        data = torch.Tensor([seq]).view(-1,1,6).to(dev)
                        predited_label = behavior(data)
                        target[4] = int(predited_label.argmax(1)+1)
            else:
                seq = seqCollector.get_sequece(seq_target)

            
            if len(target):
                bbox=bboxes[0]
                margin=10
                x1=max(0,int(float(bbox[0])*w)-margin)
                y1=max(0,int(float(bbox[1])*h)-margin)
                x2=min(w,int(float(bbox[2])*w)+margin)
                y2=min(h,int(float(bbox[3])*h)+margin)
                print(x1,y1,x2,y2,im0.shape)
                img_roi=im0[y1:y2,x1:x2].copy()
                breed = predict_breed(img_roi,breed_clf)
                print('breed :',dog_breeds[breed])
            
            # breed = predict_breed(img_roi,breed_clf)
            msgs =''
            if len(target):
                msgs="{0:0.4f},{1:0.4f},{2:0.4f},{3:0.4f},{4},{5}".format(target[0],target[1],target[2],target[3],target[4],breed)+'!'

            send_msg_to(msgs,msg_client)
            

            # send_image_to(image_padding(img_roi,(128,128)),cam_client,dsize=(640, 480))
            dt = time.time()-t