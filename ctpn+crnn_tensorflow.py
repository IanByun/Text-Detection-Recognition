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
CRNN_DIR = os.path.join(ROOT_DIR, 'CRNN_Tensorflow')

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
ctpn_config.gpu_options.allow_growth = True
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
        
        ctpn_sess.close()
        
    os.chdir(ROOT_DIR)
    return boxes

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
한 스크립트 내에서 여러 텐서플로 모델을 복원하고 실행하기 위해 다음을 참고하여 작업함
https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

os.chdir(CRNN_DIR)
import tools.demo_shadownet as crnn
import crnn_model.crnn_model as crnn_model
crnn_weights_path = 'model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999'

crnn_graph = tf.Graph()
with crnn_graph.as_default():
    crnn_net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        crnn_inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')
        crnn_net_out = crnn_net.build_shadownet(inputdata=crnn_inputdata)

crnn_decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=crnn_net_out, sequence_length=25*np.ones(1), merge_repeated=False)
crnn_decoder = crnn.data_utils.TextFeatureIO()

# config tf session
crnn_sess_config = tf.ConfigProto()
crnn_sess_config.gpu_options.allow_growth = True
crnn_sess = tf.Session(config=crnn_sess_config)
with crnn_sess.as_default():
    with crnn_graph.as_default():
        # config tf saver
        crnn_saver = tf.train.Saver()
        crnn_sess = tf.Session(config=crnn_sess_config)
        crnn_saver.restore(sess=crnn_sess, save_path=crnn_weights_path)
    
def crnn(cv_images):
    texts=[]
    os.chdir(CRNN_DIR)
    with crnn_sess.as_default():
        for image in cv_images:
            image = cv2.resize(image, (100, 32))
            image = np.expand_dims(image, axis=0).astype(np.float32)
            
            preds = crnn_sess.run(crnn_decodes, feed_dict={crnn_inputdata: image})
            preds = crnn_decoder.writer.sparse_tensor_to_str(preds[0])
            
            texts.append(preds[0])
        crnn_sess.close()
        
    os.chdir(ROOT_DIR)
    return texts

os.chdir(ROOT_DIR)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def infer_box_test(img):
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

def visualize_text_boxes(img, boxes, texts):
    img = img.copy()
    color = (0, 255, 0)
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 3)
        cv2.putText(img, texts[i], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


if __name__ == '__main__':
    img_path = 'a.jpg'
    img = cv2.imread(img_path)

    boxes, texts = infer_box_test(img)
    img = visualize_text_boxes(img, boxes, texts)

    cv2.imshow("img", img)
    cv2.waitKey()
    