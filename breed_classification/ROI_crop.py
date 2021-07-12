import cv2
import os

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # refPt와 cropping 변수를 global로 만듭니다.
    global refPt, cropping

    # 왼쪽 마우스가 클릭되면 (x, y) 좌표 기록을 시작하고
    # cropping = True로 만들어 줍니다.
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 왼쪽 마우스 버튼이 놓여지면 (x, y) 좌표 기록을 하고 cropping 작업을 끝냅니다.
    # 이 때 crop한 영역을 보여줍니다.
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        # ROI 사각형을 이미지에 그립니다.
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

        


dataset_path="./data/chihuahua/"
save_path="./data/crop/"

name='chihuahua'

# print(os.listdir(dataset_path))
total=len(os.listdir(dataset_path))
for n, img_path in enumerate(os.listdir(dataset_path)):
    print('image :',img_path)

    image = cv2.imread(dataset_path+img_path)

    h_,w_=image.shape[:2]
    image=cv2.resize(image, (640,480), interpolation=cv2.INTER_LINEAR)
    h,w=image.shape[:2]
    # 원본 이미지를 clone 하여 복사해 둡니다.
    clone = image.copy()
    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # cv2.setMouseCallback("image", click_and_crop)


    """
    키보드에서 다음을 입력받아 수행합니다.
    - q : 작업을 끝냅니다.
    - r : 이미지를 초기화 합니다.
    - c : ROI 사각형을 그리고 좌표를 출력합니다.
    """
    i=0
    while True:
        # 이미지를 출력하고 key 입력을 기다립니다.
        cv2.imshow("image", image)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z"):
            image = clone.copy()


        elif key == ord(" "):
            if len(refPt) == 2:
                roi = clone[max(0,refPt[0][1]) : min(h,refPt[1][1]), max(0,refPt[0][0]) : min(w,refPt[1][0])]
                roi=cv2.resize(roi, (w_,h_), interpolation=cv2.INTER_LINEAR)
                print(save_path+name+str(i)+img_path)
                cv2.imwrite(save_path+name+str(i)+img_path, roi)
                cv2.waitKey(10)
                i+=1
                break

        elif key == ord("v"):
            if len(refPt) == 2:
                roi = clone[max(0,refPt[0][1]) : min(h,refPt[1][1]), max(0,refPt[0][0]) : min(w,refPt[1][0])]
                roi=cv2.resize(roi, (w_,h_), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(save_path+name+str(i)+img_path, roi)
                print(save_path+name+str(i)+img_path, "saved\n")
                cv2.waitKey(10)
                image = clone.copy()
                i+=1
                

        elif key == ord("x"):
            break

        elif key == ord("c"):
            cv2.imwrite(save_path+"croped_"+str(i)+img_path, image)
            break
    
    i+=1
    print(n," / ",total)

# 모든 window를 종료합니다.
cv2.destroyAllWindows()