#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import win_unicode_console
win_unicode_console.enable()

import os
import sys
import numpy as np
import cv2
import tensorflow as tf

ROOT_DIR = os.getcwd()
CTPN_DIR = os.path.join(ROOT_DIR, 'text-detection-ctpn')
CRNN_DIR = os.path.join(ROOT_DIR, 'crnn.pytorch')

sys.path.append(ROOT_DIR)
sys.path.append(CTPN_DIR)
sys.path.append(CRNN_DIR)

os.chdir(CTPN_DIR)
import ctpn.demo as ctpn
from ctpn.demo import resize_im
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.fast_rcnn.test import test_ctpn
from lib.text_connector.detectors import TextDetector
ctpn.cfg_from_file('ctpn/text.yml')

ctpn_graph = tf.Graph()
with ctpn_graph.as_default():
    # load network
    ctpn_net = ctpn.get_network("VGGnet_test")

# init session
ctpn_config = tf.ConfigProto(allow_soft_placement=True)
ctpn_config.gpu_options.allow_growth = True #preollocate 하지 않아 gpu 메모리 사용량을 줄인다고 함
ctpn_sess = tf.Session(config=ctpn_config, graph=ctpn_graph)
with ctpn_sess.as_default():
    with ctpn_graph.as_default():
        # load model
        ctpn_saver = tf.train.Saver()
        ctpn_ckpt = tf.train.get_checkpoint_state(ctpn.cfg.TEST['checkpoints_path'])
        ctpn_saver.restore(ctpn_sess, ctpn_ckpt.model_checkpoint_path)

def ctpn(cv_image):
    os.chdir(CTPN_DIR)
    with ctpn_sess.as_default():
        img = cv_image
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(ctpn_sess, ctpn_net, img)
        
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        boxes[:, 0:8] /= scale
        
    os.chdir(ROOT_DIR)
    return boxes

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DLL load failed error when import torch-> do the below and run from cmd not powershell
https://github.com/pytorch/pytorch/issues/4518#issuecomment-384202353

Calling two deep network models exceeds even the 11GB GPU memory of 1080Ti->
https://stackoverflow.com/questions/47086338/running-out-of-memory-during-evaluation-in-pytorch
https://github.com/tensorflowkorea/tensorflow-kr/blob/master/g3doc/how_tos/using_gpu/index.md#gpu-메모리-증가-허용하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

os.chdir(CRNN_DIR)
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
crnn_model_path = 'data/crnn.pth'
crnn_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

crnn_model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    crnn_model = crnn_model.cuda()
print('loading pretrained model from %s' % crnn_model_path)

crnn_model.load_state_dict(torch.load(crnn_model_path))
crnn_converter = utils.strLabelConverter(crnn_alphabet)
crnn_transformer = dataset.resizeNormalize((100, 32))

def crnn(cv_images):
    texts = []
    os.chdir(CRNN_DIR)

    for image in cv_images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert('L')
        image = crnn_transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        with torch.no_grad():
            crnn_model.eval()
            preds = crnn_model(image)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = crnn_converter.decode(preds.data, preds_size.data, raw=False)

        texts.append(sim_pred)

    os.chdir(ROOT_DIR)
    return texts


os.chdir(ROOT_DIR)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def infer_box_text(img):
    boxes = ctpn(img)
    boxes = [[int(k) for k in box[0:8]] for box in boxes]  # x1y1 ...
    boxes = [[min(box[0], box[2], box[4], box[6]), min(box[1], box[3], box[5], box[7]),
              max(box[0], box[2], box[4], box[6]) - min(box[0], box[2], box[4], box[6]),
              max(box[1], box[3], box[5], box[7]) - min(box[1], box[3], box[5], box[7])] for box in
             boxes]  # x, y, width, height

    patches = []
    for box in boxes:
        x, y, width, height = box
        img_patch = img[y:y + height, x:x + width]
        patches.append(img_patch)

    texts = crnn(patches)

    return boxes, texts

def visualize_text_box(img, boxes, texts):
    img = img.copy()
    color = (0, 255, 0)
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 3)
    for i, text in enumerate(texts):
        cv2.putText(img, text, (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    return img


if __name__ == '__main__':
    save_directory = 'ocr_result'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    image_directory = 'ocr_data'
    image_names = [file_name for file_name in os.listdir(image_directory) if file_name.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in image_names:
        img = cv2.imread(os.path.join(image_directory, img_name))

        boxes, texts = infer_box_text(img)
        img = visualize_text_box(img, boxes, texts)

        cv2.imshow(img_name, img)
        cv2.imwrite(os.path.join(save_directory, img_name), img)
        cv2.waitKey()

    