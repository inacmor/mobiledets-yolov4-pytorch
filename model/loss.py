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


def yolo4_loss(feats,
               yolos,
               ground_truth,
               ignore_thresh=.5,
               ):

    device = feats[0].device

    (batch_size, input_size, _, num_layers, _) = feats[0].size()

    loss = torch.tensor(0, dtype=feats[0].dtype).to(device)


    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

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

        # # calculate ignore_mask
        for b in range(feats[l].size(0)):

            t_box = ground_truth[l][b, ...].to(device)
            t_box = t_box[t_box[..., 4] == 1]
            t_box = t_box[..., 0:4]

            # #calculate ciou between predict and ground truth
            if len(t_box) == 0:
                continue
            else:
                ignore_iou = iou_ignore(pred_box[b, ...], t_box, device=device, Ciou=True)

                best_iou = ignore_iou.max(dim=-1, keepdim=True)[0].squeeze().unsqueeze(-1)

                ignore_mask[b, ...] = best_iou < ignore_thresh

        box_loss_scale = (2. - ground_truth[l][..., 2] * ground_truth[l][..., 3]).unsqueeze(-1)

        # #calculate ciou loss which is bbox regression loss
        raw_true_box = ground_truth[l][..., 0:4]
        ciou_loss, _, _, _,  = iou_loss(pred_box, raw_true_box, device=device, Ciou=True, detail=True)
        ciou_loss = object_mask * box_loss_scale * ciou_loss
        ciou_loss = torch.sum(ciou_loss) / m

        confi_loss = object_mask * bce_loss(pred_confi, true_confi) + \
                     (1 - object_mask) * ignore_mask * bce_loss(pred_confi, true_confi)

        class_loss = object_mask * bce_loss(torch.sigmoid(pred_class), true_class)

        confidence_loss = torch.sum(confi_loss) / m
        class_loss = torch.sum(class_loss) / m

        loss += ciou_loss + confidence_loss + class_loss

    return loss


