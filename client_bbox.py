import socket
import numpy as np
import time


from socket_funcs import *

# 연결할 서버(수신단)의 ip주소와 port번호
with open('AWS_IP.txt', 'r') as f:
    TCP_IP = f.readline()

TCP_PORT = 5555
msg_server = socket.socket()
msg_server.connect((TCP_IP, TCP_PORT))

while True:
    

    msg_recv=recv_msg_from(msg_server)
    print(msg_recv)
    # msgs=msg_recv.split('!')

    # bboxes=[]
    # for msg in msgs[:-1]:
    #     bbox=msg.split(',')
    #     bboxes.append(bbox)

    # print(bboxes)
    # for bbox in bboxes:
    #     if bbox[-1] != "x":
    #         x1 = float(bbox[0])*w
    #         y1 = float(bbox[1])*h
    #         x2 = float(bbox[2])*w
    #         y2 = float(bbox[3])*h
    #         cls = int(bbox[-1])
    #         plot_one_box(
    #             [x1,y1,x2,y2],
    #             img,
    #             color=colors[0],
    #             label=names[cls],
    #             line_thickness=3,
    #         )

    # dt = time.time() - start
    
    # cv2.putText(img, text="fps : {:.2f}".format(1 / dt), org=(30, 30), 
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
    #                     color=(255, 255, 0), thickness=2)

    # cv2.imshow("Original", img)
    # if cv2.waitKey(1) == 27:
    #     break
cv2.destroyAllWindows()
img_server.close()
msg_server.close()