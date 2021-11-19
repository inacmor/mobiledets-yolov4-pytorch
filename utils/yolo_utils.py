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
from utils.iou import iou_nms


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
        # #create logger
        self.__logger = logging.getLogger()
        self.__logger.setLevel(log_level)
        # #create handle，将内容写入日志文件
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # #define handle format
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

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

    device = prediction[0].device

    prediction = prediction[prediction[..., 4] >= conf_thres]

    if prediction[:, 3].size(0) == 0:
        print("no obj...")
        return [], []

    score = prediction[:, 4] * prediction[:, 5:].max(1)[0].to(device)
    # #sort
    prediction = prediction[(-score).argsort()]
    class_confs, class_preds = prediction[:, 5:].max(1, keepdim=True)
    detections = torch.cat((prediction[:, :5], class_confs.float(), class_preds.float()), 1).to(device)

    keep_boxes = []
    keep_cls = []
    while detections.size(0):
        # #get the best score，and delete it from detection
        best_box = detections[0, :4].unsqueeze(0)
        keep_boxes.append(best_box)
        keep_cls.append(detections[0, 6].unsqueeze(0))
        detections = detections[1:, ...]

        # #calculate diou, and mark which`s iou<nms_thres
        nms_mask = iou_nms(best_box, detections[:, :4]) < nms_thres
        # #keep whats iou<nms_thres
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


def box_trans(b, mode, tn, i=False):
    '''
    :param b: box(es) which needs to be transfered to xyxy or xywh
    :param mode: choose 'xyxytoxywh' or 'xywhtoxyxy'
    :param tn: choose 't'(tensor) or 'n'(numpy)
    :param i: choose False(float) or True(int)
    :return: transfered box(es)
    '''

    # #check
    if mode != 'xyxytoxywh' and mode != 'xywhtoxyxy':
        print("mode error. please set mode to 'xyxytoxywh' or 'xywhtoxyxy'...")
        return b

    if tn != 't' and tn != 'n':
        print("tn error. please set tn to 't' or 'n'...")
        return b

    if len(b[0, :]) != 4 and i == True:
        print("cant concat float array to int array...")
        return b

    if mode == 'xyxytoxywh':
        x1 = b[:, 0]
        y1 = b[:, 1]
        x2 = b[:, 2]
        y2 = b[:, 3]

        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2

        if tn == 't':
            n_b = torch.cat([x.unsqueeze(0), y.unsqueeze(0), w.unsqueeze(0), h.unsqueeze(0)], dim=0).T
            n_b = n_b.float().to(b.device)
            if i == True:
                n_b = n_b.int()
        else:
            n_b = np.stack([x, y, w, h], axis=1)
            if i == True:
                n_b = n_b.astype(np.int)

    else:

        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2]
        h = b[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        if tn == 't':
            n_b = torch.cat([x1.unsqueeze(0), y1.unsqueeze(0), x2.unsqueeze(0), y2.unsqueeze(0)], dim=0).T
            n_b = n_b.float().to(b.device)
            if i == True:
                n_b = n_b.int()
        else:
            n_b = np.stack([x1, y1, x2, y2], axis=1)
            if i == True:
                n_b = n_b.astype(np.int)

    if len(b[0, :]) != 4:
        if tn == 't':
            n_b = torch.cat([n_b.clone(), b[:, 4:]], dim=1)
            n_b = n_b.to(b.device)
        else:
            n_b = np.hstack([n_b.copy(), b[:, 4:]])

    return n_b
