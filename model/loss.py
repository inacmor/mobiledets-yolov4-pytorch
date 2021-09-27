#encoding=utf-8

'''
@Time          : 2020/12/04 13:23
@Author        : Inacmor
@File          : loss.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''


import torch.nn.functional as F
import torch
from utils.iou import iou_ignore, iou_loss
from utils.yolo_utils import box_trans


def smooth_label(ground_truth, label_smoothing):
    num_classes = float(ground_truth.size(-1))
    label_smoothing = float(label_smoothing)
    return ground_truth * (1.0 - label_smoothing) + label_smoothing / num_classes


# @torchsnooper.snoop()
def yolo4_loss(feats,
               yolos,
               ground_truth,
               anchors,
               num_classes,
               stride,
               ignore_thresh=.5,
               label_smoothing=False,
               print_loss=False):
    """
    :param feats: yolo4输出列表，分为3个feature(batchsize, outsize, outsize, anchors//3, num_bbparas+1+num_classes)
    :param yolos: 对feature解码后的输出列表
    :param ground_truth:基准值数组,分为3个gt(b, outsize, outsize, anchors//3, num_bbparas+1+num_classes)
    :param anchors: k聚类后的锚框，共9组(w, h)
    :param num_classes: 分类数
    :param stride:降维步长=[8, 16, 32]
    :param ignore_thresh: NMS使用
    :param label_smoothing: 平滑标签
    :param print_loss: 打印loss
    :return:
    """

    device = feats[0].device

    (batch_size, input_size, _, num_layers, _) = feats[0].size()
    # anchor_mask = [3, 2, 1] if num_layers == 3 else [1, 0]

    loss = torch.tensor(0, dtype=feats[0].dtype).to(device)

    ciou_ls = 0
    confi_l = 0
    cls_l = 0
    #
    i_ls = 0
    d_ls = 0
    e_ls = 0

    m = torch.tensor(batch_size, dtype=feats[0].dtype).to(device)

    for l in range(num_layers):

        object_mask = ground_truth[l][:, :, :, :, 4:5]

        pred_confi = feats[l][..., 4:5]
        pred_class = feats[l][..., 5:]

        true_confi = ground_truth[l][..., 4].clone().unsqueeze(-1)
        true_class = ground_truth[l][..., 5:].clone()

        # (m,19,19,3,4)
        pred_box = yolos[l][..., 0:4]

        ignore_mask = torch.ones_like(object_mask, dtype=torch.float).to(device)

        # 对每一张图片计算ignore_mask
        for b in range(feats[l].size(0)):

            t_box = ground_truth[l][b, ...].to(device)
            t_box = t_box[t_box[..., 4] == 1]
            t_box = t_box[..., 0:4]

            # #计算预测框与真实框的ciou
            # 计算的结果是每个pred_box和其它所有真实框的iou
            if len(t_box) == 0:
                continue
            else:
                ignore_iou = iou_ignore(pred_box[b, ...], t_box, device=device, Ciou=True)

                # print(ignore_iou.max(dim=-1, keepdim=True)[0].size())
                # 19,19,3
                best_iou = ignore_iou.max(dim=-1, keepdim=True)[0].squeeze().unsqueeze(-1)

                ignore_mask[b, ...] = best_iou < ignore_thresh

        box_loss_scale = (2. - ground_truth[l][..., 2] * ground_truth[l][..., 3]).unsqueeze(-1)

        # Calculate ciou loss as location loss
        raw_true_box = ground_truth[l][..., 0:4]
        ciou_loss, l_i, l_d, l_e,  = iou_loss(pred_box, raw_true_box, device=device, Ciou=True, detail=True)
        # ciou_loss = iou_loss(pred_box, raw_true_box, device=device, Ciou=True)
        ciou_loss = object_mask * box_loss_scale * ciou_loss
        ciou_loss = torch.sum(ciou_loss) / m
        a_closs = ciou_loss / torch.sum(object_mask)

        confi_loss = object_mask * F.binary_cross_entropy(torch.sigmoid(pred_confi),
                                                          true_confi,
                                                          reduction='none') + \
                    (1 - object_mask) * ignore_mask * F.binary_cross_entropy(torch.sigmoid(pred_confi),
                                                               true_confi,
                                                               reduction='none')

        class_loss = object_mask * F.binary_cross_entropy(torch.sigmoid(pred_class),
                                                          true_class,
                                                          reduction='none')

        confidence_loss = torch.sum(confi_loss) / m
        class_loss = torch.sum(class_loss) / m

        confi_l += confidence_loss
        cls_l += class_loss

        with torch.no_grad():
            ciou_ls += a_closs
            i_ls += torch.sum(object_mask * l_i.unsqueeze(-1)) / (m * torch.sum(object_mask))
            d_ls += torch.sum(object_mask * l_d.unsqueeze(-1)) / (m * torch.sum(object_mask))
            e_ls += torch.sum(object_mask * l_e.unsqueeze(-1)) / (m * torch.sum(object_mask))

        loss += ciou_loss + confidence_loss + class_loss

    return loss, (ciou_ls, confi_l, cls_l), (ciou_ls, i_ls, d_ls, e_ls)
