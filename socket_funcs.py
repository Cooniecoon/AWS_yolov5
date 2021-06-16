import cv2
import numpy as np
import socket
import json
import time


def get_message_codes():
    with open('message_code.json', 'r') as f:
        messages = json.load(f)
    return messages
messages=get_message_codes()

# socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    msg=messages['roger']
    sock.send(msg.encode())
    return buf

def send_image_to(img,sock,dsize):
    img=cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

    # send image to client
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode(".jpg", img, encode_param)
    data = np.array(imgencode)
    stringData = data.tobytes()

    # String 형태로 변환한 이미지를 socket을 통해서 전송
    data_len = len(stringData)

    str_data_len = str(data_len).encode().ljust(16)
    sock.send(str_data_len)
    sock.send(stringData)
    recv_check(sock)

def recv_img_from(sock):
    newbuf = sock.recv(16)
    length = newbuf.decode()
    img_data = recvall(sock, int(length))
    data = np.frombuffer(img_data, dtype="uint8")
    image = cv2.imdecode(data, 1)
    return image


def send_msg_to(msgs,sock):
    if len(msgs) !=0:
        str_data = msgs.ljust(256).encode()
    else:
        str_data='0.0000,0.0000,0.0000,0.0000,x!'.ljust(256).encode()
    sock.send(str_data)
    recv_check(sock)


def recv_msg_from(sock):
    data = sock.recv(300)
    msg = data.decode()
    sock.send(messages['roger'].encode())
    return msg


def recv_check(sock):
    while True:
        msg=sock.recv(1).decode()
        if msg==messages['roger']:
            break



## image plot ##
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)