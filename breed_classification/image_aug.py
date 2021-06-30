import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
import shutil as sh
import os
import cv2
import numpy as np


def augmentation(image, augmentation):
    # print("\n", augmentation, "\n")
    image=np.expand_dims(image, axis=0)
    print(image.shape)
    seq = iaa.Sequential(augmentation)

    image_aug = seq(images=image)
    return image_aug



data_path='..\\data\\breed_data\\'

blur = iaa.AverageBlur(k=(2, 11))  #! 2~11 random
emboss = iaa.Emboss(alpha=(1.0, 1.0), strength=(2.0, 2.0))
gray = iaa.RemoveSaturation(from_colorspace=iaa.CSPACE_BGR)
contrast = iaa.AllChannelsCLAHE(clip_limit=(10, 10), per_channel=True)
bright = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
color = iaa.pillike.EnhanceColor()
sharpen = iaa.Sharpen(alpha=(0.5, 1.0))  #! 0.5 ~ 1.0 random
edge = iaa.pillike.FilterEdgeEnhance()
flip = iaa.Fliplr(1.0)  #! 100% left & right

augmentations = [[bright], [blur]]  #! choice augmentation ##
auglist=['bright', 'blur']
# classes = listdir(data_path)
# ['Chihuahua', 'Pomeranian', 'Welsh_corgi', 'etc', 'golden_retriever']
classes=['Pomeranian', 'Welsh_corgi']
print(classes)

for breed in classes:
    images = listdir(data_path+breed)
    for image in images:
        img=cv2.imread(data_path+breed+'/'+image)
        for aug,idx in zip(augmentations,auglist):
            name=data_path+breed+'/'+idx+'_'+image
            img_aug=augmentation(img,aug)[0]
            print(name)
            # cv2.imshow('aug',img_aug)
            # cv2.waitKey(10)
            cv2.imwrite(name,img_aug)