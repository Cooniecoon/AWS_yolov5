#!/usr/bin/env python
# -*- coding: utf8 -*-
import cv2
import os
from os import listdir
from xml.etree import ElementTree
from lxml import etree

ENCODE_METHOD = "utf-8"

parser = etree.XMLParser(encoding=ENCODE_METHOD)

xml_path='./data/Annotation/'
img_path='./data/Images/'
crop_path='./data/crop_images/'

folders=os.listdir(xml_path)

for folder in folders:
    for file in os.listdir(xml_path+folder):
        i=0
        img_file_path=img_path+folder+'/'+file.split('.')[0]
        img=cv2.imread(img_file_path+'.jpg')
        H,W=img.shape[:2]
        xmltree = ElementTree.parse(xml_path+folder+'/'+file, parser=parser).getroot()
        # print(file)

        for object_iter in xmltree.findall("object"):
            bndbox = object_iter.find("bndbox")
            xmin = max(int(bndbox.find("xmin").text),0)
            ymin = max(int(bndbox.find("ymin").text),0)
            xmax = min(int(bndbox.find("xmax").text),W)
            ymax = min(int(bndbox.find("ymax").text),H)
            folder_name='-'.join(' '.join(folder.split('-')[1:]).split(' '))
            crop_img=img[ymin:ymax,xmin:xmax]
            new_path=crop_path+folder_name+'/'+file.split('.')[0]+'_'+str(i)+'.jpg'
            cv2.imwrite(new_path,crop_img)
            print(crop_path+folder_name+'/'+file.split('.')[0]+'_'+str(i)+'.jpg')
            i+=1