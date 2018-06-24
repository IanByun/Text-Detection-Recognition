import os
import sys

ROOT_DIR = os.getcwd()
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'coco-text'))
sys.path.append(os.path.join(ROOT_DIR, 'train2014'))
sys.path.append(os.path.join(ROOT_DIR, 'test2014'))
sys.path.append(os.path.join(ROOT_DIR, 'val2014'))

import coco_text
ct = coco_text.COCO_Text('coco-text/COCO_Text.json')
imgs = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible'),('class','machine printed')])
anns = ct.getAnnIds(imgIds=ct.val, 
                        catIds=[('legibility','legible'),('class','machine printed')], 
                        areaRng=[0,200])

dataType='train2014'

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# get all images containing at least one instance of legible text# get a 
imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible')])

# pick one at random
img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

import cv2
img_path = '%s/%s'%(dataType,img['file_name'])
I = cv2.imread(img_path)
plt.figure()
plt.imshow(I)
plt.show()

# load and display text annotations
annIds = ct.getAnnIds(imgIds=img['id'])
anns = ct.loadAnns(annIds)
ct.showAnns(anns)
plt.imshow(I)
plt.show()


