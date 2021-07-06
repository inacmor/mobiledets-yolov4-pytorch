#encoding=utf-8

'''
@Time          : 2020/12/22 15:59
@Author        : Inacmor
@File          : detect.py
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
import matplotlib.pyplot as plt

from model.yolo4 import YOLO4
from config.yolov4_config import TRAIN, MODEL, VAL, DETECT
from utils.yolo_utils import get_classes, get_anchors, non_max_suppression
from torch.utils.data.dataloader import DataLoader
from dataset.datasets import DETECT_IMG
from cal_camera_cors import resize

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    batch_size = DETECT["BATCH_SIZE"]
    anchors = get_anchors(DETECT["ANCHOR_PATH"]).to(device)
    strides = torch.tensor(DETECT["STRIDES"]).to(device)
    # train_img_size = TRAIN["TRAIN_IMG_SIZE"]
    class_names = get_classes(DETECT["CLASS_PATH"])
    num_classes = len(class_names)
    detect_img_size = DETECT["DETECT_SIZE"]
    img_path = DETECT["IMG_PATH"]

    weights_path = DETECT["WEIGHT_PATH"]
    conf_thres = DETECT["CONF_THRES"]
    nms_thres = DETECT["NMS_THRES"]

    model = YOLO4(batch_size=1,
                  num_classes=num_classes,
                  num_bbparas=4,
                  anchors=anchors,
                  stride=strides,
                  freeze=False,
                  inference=True
                  ).to(device)
    # model = Build_Model(weight_path=weights_path).to(device)

    root_path = './weights/output/'
    w_list = os.listdir(root_path)

    for weights in w_list:

        model.load_state_dict(torch.load(weights), True)
        print(root_path+weights)

        model.eval()

        img = cv2.imread('./devs/IMG_5428.JPG')
        img = resize(img, detect_img_size)

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()

        _, prediction = model(img)

        boxes, classes = non_max_suppression(prediction, conf_thres, nms_thres)

        detected_img = cv2.imread(img_path)

        w = detected_img.shape[0]
        h = detected_img.shape[1]

        boxes[..., 0] *= w
        boxes[..., 1] *= h
        boxes[..., 2] *= w
        boxes[..., 3] *= h
        boxes = boxes[..., :4].int()
        boxes = boxes.int().cpu().numpy()

        for i in range(len(boxes)):
            detected_img = cv2.rectangle(detected_img.copy(),
                                         (boxes[i, 0], boxes[i, 1]),
                                         (boxes[i, 2], boxes[i, 3]),
                                         (0, 255, 0), 2)
            # detected_img = cv2.rectangle(detected_img,
            #                              (test01_box[0], test01_box[1]),
            #                              (test01_box[2], test01_box[3]),
            #                              (0, 0, 255), 2)

        cv2.imwrite('./devs/xxx.jpg', detected_img)
        # cv2.imshow('xxx', detected_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

