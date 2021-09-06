#encoding=utf-8

'''
@Time          : 2020/12/04 13:23
@Author        : Inacmor
@File          : yolo4_loss.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import numpy as np
import torch
import logging
import time
from utils.iou import iou_nms
import math


def get_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    cls_names = fp.read().split("\n")
    return cls_names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_anchors(anchor_path):

    anchors = []
    with open(anchor_path, "r") as a:
        lines = a.readline()
        for i in lines.split():
            for j in i.split(','):
                anchors.append(float(j.rstrip()))

    anchors = np.array(anchors).reshape(-1, 2)
    anchors = torch.tensor(anchors).unsqueeze(0).view(3, 3, 2).float()

    return anchors


class Logger(object):
    def __init__(self, log_file_name, log_level=logging.DEBUG):
        # #创建一个logger
        self.__logger = logging.getLogger()
        self.__logger.setLevel(log_level)
        # #创建一个handle，将内容写入日志文件
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # #定义handle格式
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # #给handle添加内容
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def generate_val(path, train_path, val_path, val_index, epochs):

    with open(path) as f:
        lines = f.readlines()

    sum = len(lines)

    seed = np.random.randint(0, epochs)
    np.random.seed(seed)
    np.random.shuffle(lines)

    with open(val_path, "w") as f1:
        f1.truncate(0)

        for l in range(int(len(lines) * val_index)):
            f1.write(lines[l])
        f1.close()

    with open(train_path, "w") as f2:
        f2.truncate(0)
        for l in range(int(len(lines) * (1 - val_index)) + 1):
            f2.write(lines[sum - l - 1])
        f2.close()


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    '''
    :param prediction: yolov4输出结果，包含3维度的列表
    :param conf_thres:
    :param nms_thres:
    :return:
    '''

    device = prediction[0].device

    prediction = prediction[prediction[..., 4] >= conf_thres]

    if prediction[:, 3].size(0) == 0:
        print("未检测到目标...")
        return [], []

    # Object confidence times class confidence
    score = prediction[:, 4] * prediction[:, 5:].max(1)[0].to(device)
    # 由小到大排序
    prediction = prediction[(-score).argsort()]
    class_confs, class_preds = prediction[:, 5:].max(1, keepdim=True)
    # #class_confs为类预测中最大的概率
    # #class_preds为最大概率的索引
    detections = torch.cat((prediction[:, :5], class_confs.float(), class_preds.float()), 1).to(device)

    # Perform non-maximum suppression
    keep_boxes = []
    keep_cls = []
    while detections.size(0):
        # #提取最优score，并从detection移除之
        best_box = detections[0, :4].unsqueeze(0)
        keep_boxes.append(best_box)
        keep_cls.append(detections[0, 6].unsqueeze(0))
        detections = detections[1:, ...]

        # #计算与最大值的iou，并标记iou<nms_thres
        nms_mask = iou_nms(best_box, detections[:, :4]) < nms_thres
        # #保留iou<nms_thres
        detections = torch.masked_select(detections, nms_mask.squeeze().unsqueeze(-1).repeat(1, 7))
        detections = detections.view(-1, 7)

    if keep_boxes:
        keep_boxes = torch.cat(keep_boxes, dim=0)

        x = keep_boxes[:, 0].unsqueeze(-1)
        y = keep_boxes[:, 1].unsqueeze(-1)
        w = keep_boxes[:, 2].unsqueeze(-1)
        h = keep_boxes[:, 3].unsqueeze(-1)

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        keep_boxes[:, 0:4] = torch.cat([x1, y1, x2, y2], dim=-1)
        keep_cls = torch.cat(keep_cls, dim=0).unsqueeze(-1)

    return keep_boxes, keep_cls


def xyxy2xywh(boxes):
    if len(boxes) == 0:
        return boxes
    else:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        x = x1 + w / 2
        y = y1 + h / 2

        return np.stack((x, y, w, h), axis=-1)


if __name__ == "__main__":

    c = get_classes('./test/classes.txt')
    print(c)


