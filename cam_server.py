import socket
import numpy as np
import time

from socket_funcs import *

# 수신에 사용될 내 ip와 내 port번호
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

while True:
    start = time.time()
    
    image = recv_img_from(cam_client)

    # image process

    send_image_to(image,cam_client,dsize=(640, 480))

    dt = time.time() - start
    print("fps : {:.2f}".format(1 / dt))

cam_client.close()

