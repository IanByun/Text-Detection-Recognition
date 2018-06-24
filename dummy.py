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

import coco_text
ct = coco_text.COCO_Text(os.path.join(TEXT_DIR, 'COCO_Text.json'))

import coco_evaluation
def eval_on_coco(result_json, imgs_used):
    our_results = ct.loadRes(result_json)
    our_detections = coco_evaluation.getDetections(ct, our_results, imgs_used, detection_threshold=0.5)
    our_endToEnd_results = coco_evaluation.evaluateEndToEnd(ct, our_results, imgs_used, detection_threshold=0.5)

    coco_evaluation.printDetailedResults(ct, our_detections, our_endToEnd_results, 'our approach')


if __name__ == '__main__':
    imgs_used = ct.val
    result_json = 'COCO2014_text2017_val.json'
    eval_on_coco(result_json, imgs_used)