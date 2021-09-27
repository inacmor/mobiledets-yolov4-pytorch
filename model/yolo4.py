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
from model.loss import yolo4_loss
from model.net.head import YOLO_head
from model.net.body import YOLOBODY
from dataset.datasets import resize_image, generate_groundtruth
import cv2
import numpy as np
import time

class YOLO4(nn.Module):

    def __init__(self,
                 batch_size,
                 num_classes,
                 num_bbparas,
                 anchors,
                 stride,
                 load_pretrained=False,
                 num_index=4,
                 anchor_shape='normal',
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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    weight_path = './weights/pretrained/yolov4.pth'

    batch_size = 2
    num_classes = 20
    num_bbparas = 4
    anchors = torch.tensor([[[93, 93],
                             [95, 96],
                             [99, 98]],
                            [[102, 101],
                             [106, 106],
                             [108, 108]],
                            [[113, 112],
                             [119, 119],
                             [131, 130]]]).to(device)
    stride = torch.tensor([8., 16., 32.]).to(device)

    img = torch.rand(size=(1, 3, 416, 416)).to(device)

    model = YOLO4(batch_size, num_classes, num_bbparas, anchors, stride).to(device)
    model.load_state_dict(torch.load(weight_path), False)
    model.eval()

    start = time.time()

    f, y = model(img)
    print(f[0].size())
    print(f[1].size())
    print(f[2].size())

    # for i in range(10):
    #     c1, c2, c3 = model(img)
    #
    #     # print(c1.size())
    #     # print(c2.size())
    #     # print(c3.size())
    #     # print(c4.size())
    #     # print(c5.size())
    #     end = time.time()
    #
    #     print(end - start)






