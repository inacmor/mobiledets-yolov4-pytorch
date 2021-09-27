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
import matplotlib.pyplot as plt

from model.yolo4 import YOLO4
from config.yolov4_config import TRAIN, MODEL, VAL, DA
from utils.yolo_utils import get_classes, get_anchors, non_max_suppression
from torch.utils.data.dataloader import DataLoader
from dataset.datasets import YOLO4Dataset, get_val


def initialize_train_model(device, weights_path, freeze):

    batch_size = TRAIN["BATCH_SIZE"]
    anchors = get_anchors(MODEL["ANCHOR_PATH"]).to(device)
    strides = torch.tensor(MODEL["STRIDES"]).to(device)

    class_names = get_classes(MODEL["CLASS_PATH"])
    num_classes = len(class_names)

    model = YOLO4(batch_size=batch_size,
                  num_classes=num_classes,
                  num_bbparas=4,
                  anchors=anchors,
                  stride=strides,
                  freeze=freeze
                  ).to(device)
    # model = Build_Model(weight_path=weights_path).to(device)

    model.load_state_dict(torch.load(weights_path), True)

    model.train()

    return model
