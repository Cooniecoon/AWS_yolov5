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
from socket_funcs import *
from models.DogLSTM import *

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):

    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: 
        r = min(r, 1.0)

    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    if auto:  
        dw, dh = np.mod(dw, 32), np.mod(dh, 32) 
    elif scaleFill: 
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] 
    dw /= 2 
    dh /= 2
    if shape[::-1] != new_unpad: 
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    ) 
    return img, ratio, (dw, dh)

def preprocessing(img):
    img = letterbox(img, new_shape=(640, 640))[0]
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
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    trans=transforms.ToTensor()
    img=trans(img).to(dev)
    img = img.unsqueeze(0)
    pred = model(img)[0]
    _,predicted = torch.max(pred,dim=0)
    return predicted


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

# dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'etc', 'golden_retriever']
dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'golden_retriever']

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
    weights_fname = './weights/breedClassifier_0621_cropdata_4cls_res34.pt'
    # breed_clf = DogBreedClassificationCNN()
    print('Load Classifier model')
    breed_clf = DogBreedPretrainedResnet34()
    
    breed_clf.to(dev)
    breed_clf.load_state_dict(torch.load(weights_fname))
    breed_clf.eval()

    breeds = []
    for n in dog_breeds:
        breeds.append(rename(n))

    targetFinder = TargetFinder()


    start_time = time.time()
    with torch.no_grad():
        while True:
            t = time.time()
            im0 = recv_img_from(cam_client)
            h,w=im0.shape[:2]
            img = preprocessing(im0)
            h_,w_=640,640

            # Inference
            prediction = model(img)[0]
            prediction = non_max_suppression(prediction)
            prediction = prediction[0].cpu().numpy()
            bboxes = []
            for pred in prediction:
                if pred is not None:
                    x1 = min(1,max(0,float(pred[0]/w_)))
                    y1 = min(1,max(0,float(pred[1]/h_)))
                    x2 = min(1,max(0,float(pred[2]/w_)))
                    y2 = min(1,max(0,float(pred[3]/h_)))
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
                margin=20
                y1=min(0,int(float(bbox[1])*h)-margin)
                y2=max(h,int(float(bbox[3])*h)+margin)
                x1=min(0,int(float(bbox[1])*w)-margin)
                x2=max(w,int(float(bbox[3])*w)+margin)
                img_roi=im0[y1:y2,x1:x2].copy()
                breed = predict_breed(im0,breed_clf).cpu()
                print('breed :',dog_breeds[breed])

            msgs =''
            if len(target):
                msgs="{0:0.4f},{1:0.4f},{2:0.4f},{3:0.4f},{4},{5}".format(target[0],target[1],target[2],target[3],target[4],breed)+'!'

            send_msg_to(msgs,msg_client)
            breed = predict_breed(img_roi,breed_clf).cpu()
            print('breed :',dog_breeds[breed])
            dt = time.time()-t