import socket
import cv2
import numpy as np
import time
import random
import pafy

from socket_funcs import *


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


# url = "https://www.youtube.com/watch?v=CMqROal8Usk&t=45s" # chihuahua
# url = "https://www.youtube.com/watch?v=ihjjfV5pc98" # pomeranian
# url = "https://www.youtube.com/watch?v=LYBKgDTng1w" # golden
url = "https://www.youtube.com/watch?v=a0alQnZsKX8"


video = pafy.new(url)
best = video.getbest(preftype="mp4")

cam = cv2.VideoCapture(best.url)
# cam = cv2.VideoCapture('test_video/test_2.avi')
# cam.set(5,60)

_,img=cam.read()

with open('AWS_IP.txt', 'r') as f:
    TCP_IP = f.readline()

TCP_PORT = 6666
img_server = socket.socket()
img_server.connect((TCP_IP, TCP_PORT))


TCP_PORT = 5555
msg_server = socket.socket()
msg_server.connect((TCP_IP, TCP_PORT))


names=['jump', 'rest', 'run', 'sit', 'stand', 'walk']
# dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'etc', 'golden_retriever']
dog_breeds=['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'etc', 'golden_retriever']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
record = cv2.VideoWriter('./video/output.avi', fourcc, 30.0, (640, 480))


while True:
    start = time.time()
    _,img=cam.read()
    h,w=img.shape[:2]

    # img=img[:,:w//2].copy()
    # h,w=img.shape[:2]

    img=np.ascontiguousarray(img)

    send_image_to(img,img_server,dsize=(640, 480))

    msg_recv=recv_msg_from(msg_server)
    # print(msg_recv)
    msgs=msg_recv.split('!')

    bboxes=[]
    for msg in msgs[:-1]:
        bbox=msg.split(',')
        bboxes.append(bbox)

    print(bboxes)
    for bbox in bboxes:
        if bbox[-2] != "x":
            x1 = int(float(bbox[0])*w)
            y1 = int(float(bbox[1])*h)
            x2 = int(float(bbox[2])*w)
            y2 = int(float(bbox[3])*h)
            cls = int(bbox[-2])
            breed=int(bbox[-1])
            if cls == 3 or cls ==4:
                plot_one_box(
                    [x1,y1,x2,y2],
                    img,
                    color=colors[0],
                    label='stop',
                    line_thickness=3,
                )
            else:
                plot_one_box(
                    [x1,y1,x2,y2],
                    img,
                    color=colors[0],
                    label=names[cls],
                    line_thickness=3,
                )

            cv2.putText(img, text="Breed : {}".format(dog_breeds[breed]), org=(30, 60), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                        color=(255, 255, 0), thickness=2)

    dt = time.time() - start
    
    cv2.putText(img, text="fps : {:.2f}".format(1 / dt), org=(30, 30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                        color=(255, 255, 0), thickness=2)

    cv2.imshow("Original", img)
    
    # im0 = recv_img_from(img_server)
    # cv2.imshow("CROP", im0)

    record.write(img)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
img_server.close()
msg_server.close()
