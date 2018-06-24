#!/usr/bin/python
# -*- coding: utf-8 -*-
import win_unicode_console
win_unicode_console.enable()

import os
import sys

ROOT_DIR = os.getcwd()
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(ROOT_DIR, 'COCO')
TEXT_DIR = os.path.join(COCO_DIR, 'coco-text')

sys.path.append(COCO_DIR)
sys.path.append(TEXT_DIR)

from ctpn_crnn_pytorch import *

import coco_text
ct = coco_text.COCO_Text(os.path.join(TEXT_DIR, 'COCO_Text.json'))
imgs_val = ct.val
imgs_all = ct.imgs.keys()

dataType = 'val'
imgs_used = imgs_val

def make_result_dic(utf8_string, image_id, bbox):
    result = {
        "utf8_string": utf8_string,
        "image_id": image_id,
        "bbox": bbox
    }
    return result

def infer_on_coco():
    total_results = []
    for i, img_id in enumerate(imgs_used):
        print('...processing {0}/{1} images'.format(i + 1, len(imgs_used)))

        img = ct.loadImgs(img_id)[0]
        img_path = '%s/%s' % ('train2014', img['file_name']) # train도 val도 test도 train2014 이미지로 함
        img_path = os.path.join(COCO_DIR, img_path)

        cv_img = cv2.imread(img_path)
        boxes, texts = infer_box_text(cv_img)

        for i, box in enumerate(boxes):
            result = make_result_dic(texts[i], img_id, box)
            total_results.append(result)
    return total_results

import json
def inferred_to_json(total_results):
    json_out = 'COCO2014_Text2017_' + dataType + '.json'
    print('Json opened', json_out)

    with open(json_out, 'w', encoding='UTF-8') as json_file:
        jsonString = json.dumps(total_results, indent=4)
        json_file.write(jsonString)
    
    return json_out

import coco_evaluation
def eval_on_coco(result_json, imgs_used=imgs_used):
    our_results = ct.loadRes(result_json)
    our_detections = coco_evaluation.getDetections(ct, our_results, imgs_used, detection_threshold=0.5)
    our_endToEnd_results = coco_evaluation.evaluateEndToEnd(ct, our_results, imgs_used, detection_threshold=0.5)

    coco_evaluation.printDetailedResults(ct, our_detections, our_endToEnd_results, 'our approach')


if __name__ == '__main__':
    total_results = infer_on_coco()
    result_json = inferred_to_json(total_results)
    eval_on_coco(result_json)