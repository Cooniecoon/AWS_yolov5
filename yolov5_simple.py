#!/usr/bin/env python

import torch
import numpy as np
from numpy import random
import socket
from models.experimental import attempt_load
from utils.plots import plot_one_box
from utils.general import non_max_suppression

import cv2
import sys
import time

from socket_funcs import *

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def preprocessing(img):
    img = letterbox(img, new_shape=(640, 640))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to("cuda:0")
    img = img.half()  # if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img


def bbox_center(bbox):
    if len(bbox):
        cx = bbox[0] + (bbox[2] - bbox[0]) // 2
        cy = bbox[1] + (bbox[3] - bbox[1]) // 2
        return [cx, cy]


def calculate_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


model_path = "./weights/behavior_s_0615.pt"

with open('AWS_IP.txt', 'r') as f:
    TCP_IP = f.readline()
TCP_PORT = 6666

# TCP소켓 열고 수신 대기
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)

print('listening...')
cam_client, addr = s.accept()
print("connected")


if __name__ == "__main__":

    model = attempt_load(model_path, map_location="cuda")
    model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS
    model.half()
    names = model.module.names if hasattr(model, "module") else model.names
    print("classes : ",names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    while True:
        t = time.time()
        im0 = recv_img_from(cam_client)

        img = preprocessing(im0)

        # Inference
        prediction = model(img)[0]

        prediction = non_max_suppression(prediction)
        prediction = prediction[0].cpu().numpy()

        bboxes = []
        for pred in prediction:

            if pred is not None:
                x1 = int(pred[0])
                y1 = int(pred[1])
                x2 = int(pred[2])
                y2 = int(pred[3])
                cls = int(pred[-1])
                bboxes.append([x1, y1, x2, y2, cls])

                plot_one_box(
                    [x1, y1, x2, y2],
                    im0,
                    color=colors[0],
                    label=model.names[cls],
                    line_thickness=3,
                )
        print(bboxes)
        # send_image_to(im0,cam_client,dsize=(640, 480))