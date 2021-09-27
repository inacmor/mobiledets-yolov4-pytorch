#encoding=utf-8

'''
@Time          : 2021/09/01 10:00
@Author        : Inacmor
@File          : detect_supports.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import torch
import os
import time
import cv2
import numpy as np


from model.yolo4 import YOLO4
from config.yolov4_config import DETECT
from utils.yolo_utils import get_classes, get_anchors, non_max_suppression


def predict(img, model, device):

    detect_img_size = DETECT["DETECT_SIZE"]
    conf_thres = DETECT["CONF_THRES"]
    nms_thres = DETECT["NMS_THRES"]

    cls = get_classes('./data/classes.txt')

    new_img = cv2.resize(img, (detect_img_size, detect_img_size))
    new_img = new_img / 255.0

    new_img = torch.from_numpy(new_img).permute(2, 0, 1).unsqueeze(0).to(device).float()

    _, prediction = model(new_img)

    boxes, cls_id = non_max_suppression(prediction, conf_thres, nms_thres)

    if len(boxes):
        detected_img = img.copy()

        h = detected_img.shape[0]
        w = detected_img.shape[1]

        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h
        boxes = (boxes[..., :4].int().cpu()).numpy()

        for i in range(len(boxes)):

            c = int(cls_id[i, 0].item())

            detected_img = cv2.rectangle(detected_img,
                                         (boxes[i, 0], boxes[i, 1]),
                                         (boxes[i, 2], boxes[i, 3]),
                                         (0, 255, 0), 2)
            detected_img = cv2.putText(detected_img, cls[c],
                                       (boxes[i, 0] - 2, boxes[i, 1] - 2),
                                       cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 0, 0), 2)

        return detected_img

    else:
        print("未检测到目标...")
        return img


def initialize_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    anchors = get_anchors(DETECT["ANCHOR_PATH"]).to(device)
    strides = torch.tensor(DETECT["STRIDES"]).to(device)
    # train_img_size = TRAIN["TRAIN_IMG_SIZE"]
    class_names = get_classes(DETECT["CLASS_PATH"])
    num_classes = len(class_names)

    weights_path = DETECT["WEIGHT_PATH"]

    print("\nPerforming object detection:")

    model = YOLO4(batch_size=1,
                  num_classes=num_classes,
                  num_bbparas=4,
                  anchors=anchors,
                  stride=strides,
                  freeze=False,
                  inference=True
                  ).to(device)
    # model = Build_Model(weight_path=weights_path).to(device)

    model.load_state_dict(torch.load(weights_path), True)

    model.eval()

    return model, device

