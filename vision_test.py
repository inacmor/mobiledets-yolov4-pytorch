#encoding=utf-8

'''
@Time          : 2021/04/23 19:19
@Author        : Inacmor
@File          : vision_test.py
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


def detect_resize_image(image, size):
    '''
    :param image: 原图片
    :param boxes: (classes, x1y1x2y2) 标签框列表
    :param size: yolo输入尺寸
    :return:
    '''

    # #torch读取图片格式为(3, w, h)
    # _, iw, ih = image.shape
    # image = image.permute(1, 2, 0)
    w = size
    h = size
    image = image.numpy()
    ih, iw, _ = image.shape

    new_image = cv2.resize(image, (size, size))

    new_image = new_image / 255.0

    new_image = torch.tensor(new_image).permute(2, 0, 1).float()

    return new_image


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # device = torch.device("cpu")
    #
    # batch_size = DETECT["BATCH_SIZE"]
    # anchors = get_anchors(DETECT["ANCHOR_PATH"]).to(device)
    # strides = torch.tensor(DETECT["STRIDES"]).to(device)
    # # train_img_size = TRAIN["TRAIN_IMG_SIZE"]
    # class_names = get_classes(DETECT["CLASS_PATH"])
    # num_classes = len(class_names)
    # detect_img_size = DETECT["DETECT_SIZE"]
    # img_path = DETECT["IMG_PATH"]
    #
    # weights_path = DETECT["WEIGHT_PATH"]
    # conf_thres = DETECT["CONF_THRES"]
    # nms_thres = DETECT["NMS_THRES"]
    #
    # model = YOLO4(batch_size=1,
    #               num_classes=num_classes,
    #               num_bbparas=4,
    #               anchors=anchors,
    #               stride=strides,
    #               freeze=False,
    #               inference=True
    #               ).to(device)
    #
    # model.load_state_dict(torch.load(weights_path), True)
    #
    # model.eval()
    #
    # print("\nPerforming vison detection:")
    # capture = cv2.VideoCapture('./devs/dev_01.mp4')
    #
    # # 读取某一帧
    # ref, frame = capture.read()
    #
    # while ref:
    #
    #     # print(frame.shape)
    #     # # 格式转变，BGRtoRGB
    #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     # # 转变成Image
    #     # # frame = Image.fromarray(np.uint8(frame))
    #     #
    #     # # 进行检测
    #     # new_frame, _ = detect_resize_image(frame, 320)
    #     #
    #     # with torch.no_grad():
    #     #     _, prediction = model(new_frame)
    #     #
    #     #     boxes, classes = non_max_suppression(prediction, conf_thres, nms_thres)
    #     #
    #     # # RGBtoBGR满足opencv显示格式
    #     # # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     #
    #     # fps = (fps + (1. / (time.time() - t1))) / 2
    #
    #     cv2.imshow("video", frame)
    #     c = cv2.waitKey(1) & 0xff
    #
    # capture.release()
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture('./devs/dev_01.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
