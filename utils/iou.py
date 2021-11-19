#encoding=utf-8

'''
@Time          : 2020/12/03 09:40
@Author        : Inacmor
@File          : iou.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''


import torch
import math

def xywh_xyxy(box, mode):
    if mode == 'xywh_to_xyxy':
        x = box[:, 0]
        y = box[:, 1]
        w = box[:, 2]
        h = box[:, 3]

        # #x1
        box[:, 0] = x - w / 2
        # #y1
        box[:, 1] = y - h / 2
        # #x2
        box[:, 2] = x + w / 2
        # #y2
        box[:, 3] = y + h / 2

        return box

    elif mode == 'xyxy_to_xywh':
        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]

        w = x2 - x1
        h = y2 - y1

        x = x1 + w / 2
        y = y1 + h / 2

        box[:, 0] = x
        box[:, 1] = y
        box[:, 2] = w
        box[:, 3] = h

        return box


def cal_iou(b1, b2, device, Diou=False, Ciou=False, detail=False):

    eps = torch.tensor(1e-6).to(device)

    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    zero = torch.zeros_like(intersect_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, zero)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / torch.max(b1_area + b2_area - intersect_area, eps)

    if Diou or Ciou:
        center_distance = torch.sum((b1_xy - b2_xy).pow(2), dim=-1)
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, zero)
        enclose_diagonal = torch.sum(enclose_wh.pow(2), dim=-1).to(device)
        rd = center_distance / torch.max(enclose_diagonal, eps)

        diou = iou - rd

        if Ciou:
            nu = (4 / math.pi ** 2) * torch.pow(torch.atan(b1[..., 2] / torch.max(b1[..., 3], eps)) -
                                                torch.atan(b2[..., 2] / torch.max(b2[..., 3], eps)), 2)
            alpha = nu / (1 - iou + nu)

            if detail:
                return diou - alpha * nu, iou, rd, alpha * nu
            else:
                return diou - alpha * nu
        elif Diou:
            return diou
    else:
        return iou


def iou_ignore(b1, b2, device, Diou=False, Ciou=False):
    """
    :param b1: pred_box=(fsize, fsize, 3, 4)
    :param b2: true_box=(1, n, 4)
    :return:iou/diou/ciou
    """
    b1 = b1.unsqueeze(-2)

    b2 = b2.unsqueeze(0)

    output = cal_iou(b1, b2, device, Diou, Ciou)

    return output


def iou_loss(b1, b2, device, Diou=False, Ciou=False, detail=False):
    """
    :param b1: pred_box=(batch_size, fsize, fsize, 3, 4)
    :param b2: true_box=(batch_size, fsize, fsize, 3, 4)
    :return:iou/diou/ciou
    """

    b1 = torch.clamp(b1, -1, 1)
    b2 = torch.clamp(b2, -1, 1)

    if not detail:
        output = cal_iou(b1, b2, device, Diou, Ciou, detail)
        return (1 - output).squeeze().unsqueeze(-1)
    else:
        output, l_iou, l_d, l_c = cal_iou(b1, b2, device, Diou, Ciou, detail)
        return (1 - output).squeeze().unsqueeze(-1), l_iou, l_d, l_c


def iou_label(b1, b2, Diou=False, Ciou=False):
    '''
    :param b1:=(classes, 4)
    :param b2:=(classes, 3, 4)
    :return:
    '''

    device = b1.device

    b1 = b1.unsqueeze(1)
    output = cal_iou(b1, b2, device, Diou, Ciou)

    return output


def iou_nms(b1, b2, Diou=False):
    '''
    :param b1:=(1, 4)
    :param b2:=(all, 4)
    :return:
    '''
    device = b1.device
    Ciou = False

    b1 = b1.unsqueeze(1)
    output = cal_iou(b1, b2, device, Diou, Ciou)
    return output.squeeze()

