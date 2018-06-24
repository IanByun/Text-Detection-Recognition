#!/usr/bin/python
# -*- coding: utf-8 -*-
import win_unicode_console
win_unicode_console.enable()
import os
import sys
import numpy as np
import cv2

ROOT_DIR = os.getcwd()
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'coco-text'))
sys.path.append(os.path.join(ROOT_DIR, 'train2014'))
sys.path.append(os.path.join(ROOT_DIR, 'test2014'))
sys.path.append(os.path.join(ROOT_DIR, 'val2014'))

import coco_text
ct = coco_text.COCO_Text('coco-text/COCO_Text.json')

imgs_train = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible'),('language','english')])
imgs_val = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('language','english')])
imgs_test = ct.getImgIds(imgIds=ct.test, catIds=[('legibility','legible'),('language','english')])

def make_for_crnn(imgIds, dataType):
    csv_out = 'COCO_Text_' + dataType + '.csv'
    print('CSV opened', csv_out)
    
    with open(csv_out, 'w', encoding='UTF-8') as csv_file:
        for i, img_id in enumerate(imgIds):
            print('...processing {0}/{1} images'.format(i+1, len(imgIds)))
        
            img = ct.loadImgs(img_id)[0]
            img_path = '%s/%s'%('train2014', img['file_name']) #train도 val도 test도 train2014 이미지로 함
            img_path = os.path.join(ROOT_DIR, img_path)
            
            cv_img = cv2.imread(img_path)
            
            annIds = ct.getAnnIds(imgIds=img['id'], catIds=[('legibility','legible'),('language','english')])
            anns = ct.loadAnns(annIds)
            
            # 각 바운딩 박스마다, 새로운 이미지로 저장
            for j, annot in enumerate(anns): 
                id = annot['id']
                
                bbox = annot['bbox']
                x, y, width, height = [int(k) for k in bbox]
                #roi = cv_img[y:y+height, x:x+width]
                
                new_img_path = img['file_name'].replace('.jpg','')
                new_img_path += '_' + str(id) + '.jpg'
                new_img_path = '%s/%s/%s'%(dataType, 'crnn', new_img_path)
                new_img_path = os.path.join(ROOT_DIR, new_img_path)
                """
                if not os.path.exists(os.path.dirname(new_img_path)):
                    try:
                        os.makedirs(os.path.dirname(new_img_path))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                cv2.imwrite(new_img_path, roi)
                """
                if width >=10 and height >= 10:
                    utf8_string = annot['utf8_string']
                    utf8_string = '|'.join(utf8_string)
                    utf8_string = '|' + utf8_string + '|'
                    
                    csv_line = new_img_path + ';' + utf8_string +'\n'
                    csv_file.write(csv_line)


dataType_train = 'train2014'
dataType_val = 'val2014'
dataType_test = 'test2014'

make_for_crnn(imgs_train, dataType_train)
make_for_crnn(imgs_val, dataType_val)
#make_for_crnn(imgs_test, dataType_test) #no public annotations
