import socket
import cv2
import numpy as np
import time
import random
import pafy

from socket_funcs import *


url = "https://www.youtube.com/watch?v=PrUU3bd0MgI"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

cam = cv2.VideoCapture(best.url)

_,img=cam.read()


# 연결할 서버(수신단)의 ip주소와 port번호
with open('AWS_IP.txt', 'r') as f:
    TCP_IP = f.readline()

TCP_PORT = 6666
img_server = socket.socket()
img_server.connect((TCP_IP, TCP_PORT))

TCP_PORT = 5555
msg_server = socket.socket()
msg_server.connect((TCP_IP, TCP_PORT))

names=['jump', 'rest', 'run', 'sit', 'stand', 'walk']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
while True:
    start = time.time()
    _,img=cam.read()
    h,w=img.shape[:2]

    send_image_to(img,img_server,dsize=(320, 320))

    # img_recv=recv_img_from(img_server)

    msg_recv=recv_msg_from(msg_server)
    # print(msg_recv)
    msgs=msg_recv.split('!')

    bboxes=[]
    for msg in msgs[:-1]:
        bbox=msg.split(',')
        bboxes.append(bbox)

    print(bboxes)
    for bbox in bboxes:
        if bbox[-1] != "x":
            x1 = float(bbox[0])*w
            y1 = float(bbox[1])*h
            x2 = float(bbox[2])*w
            y2 = float(bbox[3])*h
            cls = int(bbox[-1])
            plot_one_box(
                [x1,y1,x2,y2],
                img,
                color=colors[0],
                label=names[cls],
                line_thickness=3,
            )

    dt = time.time() - start
    
    cv2.putText(img, text="fps : {:.2f}".format(1 / dt), org=(30, 30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                        color=(255, 255, 0), thickness=2)

    cv2.imshow("Original", img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
img_server.close()
msg_server.close()