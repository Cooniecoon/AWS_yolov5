import cv2
import numpy as np
import time
import pafy



# url = "https://www.youtube.com/watch?v=CMqROal8Usk&t=45s" # chihuahua
url = "https://www.youtube.com/watch?v=ihjjfV5pc98" # pomeranian
# url = "https://www.youtube.com/watch?v=LYBKgDTng1w" # golden
# url = "https://www.youtube.com/watch?v=a0alQnZsKX8" # welsh


video = pafy.new(url)
best = video.getbest(preftype="mp4")

cam = cv2.VideoCapture(best.url)

_,img=cam.read()


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
record = cv2.VideoWriter('./video/output.avi', fourcc, 30.0, (640, 480))

count=0

while True:
    _,img=cam.read()
    h,w=img.shape[:2]

    img=np.ascontiguousarray(img)

    if(int(cam.get(1)) % 45 == 0): 
        print('Saved frame number : ' + str(int(cam.get(1))))
        cv2.imwrite(f"data/frame{count}.jpg", img)
        count += 1

cv2.destroyAllWindows()
img_server.close()
msg_server.close()
