#encoding=utf-8

'''
@Time          : 2020/11/30 08:30
@Author        : Inacmor
@File          : yolo4.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch.nn as nn
import torch
from model.net.head import YOLO_head
from model.net.body import YOLOBODY
import time

class YOLO4(nn.Module):

    def __init__(self,
                 batch_size,
                 num_classes,
                 num_bbparas,
                 anchors,
                 stride,
                 freeze=False,
                 inference=False):
        super(YOLO4, self).__init__()
        self.inference = inference

        self.yolobody = YOLOBODY(in_channels=256, anchors=len(anchors) * 3, num_bbparas=num_bbparas, num_classes=num_classes, freeze=freeze)
        self.yolo_1 = YOLO_head(batch_size, num_classes, anchors[2], stride[2], num_bbparas, inference)
        self.yolo_2 = YOLO_head(batch_size, num_classes, anchors[1], stride[1], num_bbparas, inference)
        self.yolo_3 = YOLO_head(batch_size, num_classes, anchors[0], stride[0], num_bbparas, inference)

    def forward(self, input):

        output = self.yolobody(input)

        feat1, yolo_1 = self.yolo_1(output[0])
        feat2, yolo_2 = self.yolo_2(output[1])
        feat3, yolo_3 = self.yolo_3(output[2])

        if self.inference:
            return [feat1, feat2, feat3], torch.cat([yolo_1, yolo_2, yolo_3], dim=0)
        else:
            return [feat1, feat2, feat3], [yolo_1, yolo_2, yolo_3]







