#encoding=utf-8

'''
@Time          : 2020/11/28 12:40
@Author        : Inacmor
@File          : head.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''


import torch
import torch.nn as nn

# @torchsnooper.snoop()
class YOLO_head(nn.Module):

    def __init__(self,
                 batch_size,
                 num_classes,
                 anchors_per_feat,
                 stride,
                 num_bbparas,
                 inference=False
                 ):
        super(YOLO_head, self).__init__()

        # #index为框参数，直立矩形框默认xywh4个
        self.batch_size = batch_size
        self.num_bbparas = num_bbparas
        self.num_classes = num_classes
        self.anchors = anchors_per_feat
        self.num_anchors = len(anchors_per_feat)
        self.stride = stride
        self.inference = inference

    def forward(self, feature):

        output_size = feature.shape[-1]
        batch_size = feature.size(0)
        feature = feature.view(batch_size,
                               self.num_anchors,
                               self.num_bbparas + 1 + self.num_classes,
                               output_size,
                               output_size
                               ).permute(0, 3, 4, 1, 2)
        # feature = feature.view(self.batch_size,
        #                        self.num_anchors,
        #                        self.num_bbparas + 1 + self.num_classes,
        #                        output_size,
        #                        output_size
        #                        ).permute(0, 3, 4, 1, 2)

        feature_de = self.decoding_normal(feature.clone())

        return feature, feature_de

    def decoding_normal(self, feature):

        batch_size = feature.size(0)
        output_size = feature.shape[1]

        device = feature.device
        anchors = (self.anchors / self.stride).to(device)

        # #slice feature
        raw_dxy = feature[:, :, :, :, 0:2]
        raw_dwh = feature[:, :, :, :, 2:4]
        raw_confi = feature[:, :, :, :, 4:5]
        raw_class = feature[:, :, :, :, 5:]

        # apply grid_x, grid_y for computing tx ty
        grid_x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_y = grid_x.T
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = (grid_xy.
                   unsqueeze(0).
                   unsqueeze(3).
                   repeat(batch_size, 1, 1, self.num_anchors, 1).
                   float().
                   to(device)
                   )

        pred_xy = (torch.sigmoid(raw_dxy) + grid_xy) / output_size
        pred_wh = (torch.exp(raw_dwh) * anchors / output_size).float()
        # pred_xy = (torch.sigmoid(raw_dxy) + grid_xy) * self.stride
        # pred_wh = (torch.exp(raw_dwh) * anchors * self.stride).float()
        pred_confi = torch.sigmoid(raw_confi)
        pred_class = torch.sigmoid(raw_class)
        pred_boxes = torch.cat([pred_xy, pred_wh, pred_confi, pred_class], dim=-1)

        if not self.inference:
            return pred_boxes
        else:
            if batch_size == 1:
                return pred_boxes.view(-1, 4 + 1 + self.num_classes)
            else:
                return pred_boxes.view(batch_size, -1, 4 + 1 + self.num_classes)

