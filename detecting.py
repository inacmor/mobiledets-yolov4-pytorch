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

def detect_images(img,
                  model,
                  device,
                  detect_img_size,
                  conf_thres,
                  nms_thres):

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
        boxes = boxes[..., :4].int()

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

    list = os.listdir('./data/Test_A/')

    for l in range(len(list)):

        if l >= 30:
            break

        path = './data/Test_A/' + list[l]
        print(path)

        img = cv2.imread(path)
        print(img.shape)

        detected_img = detect_images(img, model, device, detect_img_size, conf_thres, nms_thres)

        cv2.imwrite('./ssss/' + str(l) + '.jpg', detected_img)




