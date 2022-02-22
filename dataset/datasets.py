# encoding=utf-8

'''
@Time          : 2020/12/07 17:14
@Author        : Inacmor
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.iou import iou_label
from PIL import Image
from utils.yolo_utils import generate_val, box_trans
from dataset.data_augmentation import *


def resize_image(image, boxes, size):
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
    ih, iw, _ = image.shape
    # new_image = cv2.resize(image, (w, h))
    # #normalized (0, 1)

    new_image = cv2.resize(image, (size, size))

    # image_paded[dh: size + dh, dw: size + dw, :] = new_image

    w_scale = w / iw
    h_scale = h / ih

    if boxes == []:
        pass
    else:

        boxes[:, 1] *= w_scale
        boxes[:, 2] *= h_scale
        boxes[:, 3] *= w_scale
        boxes[:, 4] *= h_scale

    return new_image, boxes


def generate_groundtruth(labels, device, input_size, anchors, stride, num_classes, anchor_thre=0.3):
    '''
    :param true_boxes:=list 包含每个batch中，由标签转化的框参数(classes, x1, y1, x2, y2)
    :param input_size: yolo网络输入尺寸
    :param anchors:=(3, 3, 2) 聚类的锚框尺寸
    :param stride:=(3) 降维步长
    :param num_classes:类别数量
    :return:
    '''

    with torch.no_grad():
        num_anchors = len(anchors)
        anchors = anchors.float()
        anchors = anchors / torch.sort(stride, descending=True)[0].unsqueeze(-1).unsqueeze(-1)
        batch_size = len(labels)

        output_size = (input_size // stride.unsqueeze(0)).clone()
        output_size = output_size.int()

        ground_truth1 = torch.zeros(size=(batch_size,
                                          output_size[0, 2],
                                          output_size[0, 2],
                                          num_anchors,
                                          4 + 1 + num_classes)).to(device)
        ground_truth2 = torch.zeros(size=(batch_size,
                                          output_size[0, 1],
                                          output_size[0, 1],
                                          num_anchors,
                                          4 + 1 + num_classes)).to(device)
        ground_truth3 = torch.zeros(size=(batch_size,
                                          output_size[0, 0],
                                          output_size[0, 0],
                                          num_anchors,
                                          4 + 1 + num_classes)).to(device)

        ground_truth = (ground_truth1, ground_truth2, ground_truth3)

        for b in range(batch_size):

            true_boxes = labels[b].view(-1, 5).to(device)
            s = (true_boxes.sum(dim=1) != 1).unsqueeze(-1).repeat(1, 5)
            true_boxes = torch.masked_select(true_boxes, s).view(-1, 5)

            true_box = box_trans(true_boxes[:, 1:5], mode='xyxytoxywh', tn='t')

            # print(true_box[:, 0].size())
            # print(stride.size())

            x_index = (true_box[:, 0].unsqueeze(-1) / stride).int().to(device)
            y_index = (true_box[:, 1].unsqueeze(-1) / stride).int().to(device)

            # # #xyxy_to_xywh
            # x1 = true_boxes[:, 1].unsqueeze(-1)
            # y1 = true_boxes[:, 2].unsqueeze(-1)
            # x2 = true_boxes[:, 3].unsqueeze(-1)
            # y2 = true_boxes[:, 4].unsqueeze(-1)
            #
            # true_w = torch.abs(x2 - x1).to(device)
            # true_h = torch.abs(y2 - y1).to(device)
            # true_x = (x1 + true_w / 2).to(device)
            # true_y = (y1 + true_h / 2).to(device)
            #
            # true_box = torch.cat([true_x, true_y, true_w, true_h], dim=-1).float()

            # x_index = (true_x / stride).int().to(device)
            # y_index = (true_y / stride).int().to(device)

            for l in range(num_anchors):

                # #(classes, 3, 1+1+2)
                anchors_xywh = torch.cat([x_index[:, l].unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 1) + 0.5,
                                          y_index[:, l].unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 1) + 0.5,
                                          anchors[2 - l].repeat(len(x_index), 1, 1)
                                          ],
                                         dim=-1
                                         )

                iou = iou_label(true_box / stride[l], anchors_xywh, Ciou=True)
                iou_mask = (iou > anchor_thre)

                for i in range(len(true_box)):

                    if iou_mask[i, :].sum() == 0:
                        continue

                    else:
                        ground_truth[2 - l][b, y_index[i, l], x_index[i, l], iou_mask[i, :], 0:4] = true_box[i,
                                                                                                    0:4] / input_size

                        # #配置confidence
                        ground_truth[2 - l][b, y_index[i, l], x_index[i, l], iou_mask[i, :], 4] = 1.

                        # #配置classes
                        cls = true_boxes[i, 0]
                        cls_pre = torch.zeros(size=(1, int(cls)), dtype=torch.float)
                        cls_pas = torch.zeros(size=(1, num_classes - int(cls) - 1), dtype=torch.float)
                        cls_prob = torch.tensor([1.], dtype=cls_pre.dtype).unsqueeze(0)
                        cls_sum = torch.cat([cls_pre, cls_prob, cls_pas], dim=-1).to(device)

                        ground_truth[2 - l][b, y_index[i, l], x_index[i, l], iou_mask[i, :], 5:] = cls_sum

                    # print("~~~~~~~~~~~~~")
                    # print(i)
                    # print(l)
                    # print(x_index[i, l])
                    # print(y_index[i, l])
                    # print(iou_mask[i, :])
                    # print(true_box[i, 0:4] / stride[l])
                    # print(input_size)
                    # print(ground_truth[2 - l][b, y_index[i, l], x_index[i, l], iou_mask[i, :], :])
                    # print("~~~~~~~~~~~~~")

    return ground_truth

# @torchsnooper.snoop()
class YOLO4Dataset(Dataset):
    def __init__(self, train_path, img_size, aug_paras, data_aug=True, check=False):

        self.train_path = train_path
        self.img_paths = []
        self.labels = []
        self.label_length = []
        self.check = check

        with open(train_path, "r") as p:
            while True:
                label = []
                lines = p.readline()
                if not lines:
                    break

                ori = np.zeros(shape=[1, 5]) - 1
                for i, j in enumerate(lines.split()):
                    print(j)
                    if i == 0:
                        self.img_paths.append(j)
                    else:
                        js = j.split(',')
                        label = np.zeros(shape=[1, 5]) - 1
                        label[0, :] = [float(js[4]), float(js[0]), float(js[1]), float(js[2]), float(js[3])]
                        ori = np.concatenate((ori, label), axis=-1)
                # ori = np.delete(ori, 0, 0)
                ori = ori[0, 5:]
                self.label_length.append(ori.shape[0] / 5.)
                self.labels.append(ori)

        self.img_size = img_size
        self.aug_paras = aug_paras
        self.data_aug = data_aug
        self.show_id = 0

    def __getitem__(self, index):

        img_path = self.img_paths[index % len(self.img_paths)].rstrip()
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # img = transforms.ToTensor()(cv2.imread(img_path))
        # #使用cv2读取图像数据可能会报错

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        if self.labels:
            boxes = self.labels[index].reshape(-1, 5)
            img, boxes = resize_image(np.copy(img), np.copy(boxes), self.img_size)
            if self.data_aug:
                img, boxes = flip(np.copy(img), np.copy(boxes), self.aug_paras["flip_mode"])
                # img, boxes = rotate(np.copy(img), np.copy(boxes))
                img, boxes = random_crop_resize(np.copy(img), np.copy(boxes))
                img = convert(np.copy(img),
                              self.aug_paras["contrast"],
                              self.aug_paras["bri_low"],
                              self.aug_paras["bri_up"])
                img = gasuss_noise(np.copy(img))

                self.show_id += 1

                if self.check:
                    im = img.copy()
                    box = boxes.astype('int32')

                    for l in range(len(box)):
                        cv2.rectangle(im, (box[l, 1], box[l, 2]), (box[l, 3], box[l, 4]), (0, 255, 0), 2)
                        cv2.circle(im, (box[l, 1], box[l, 2]), 3, (255, 0, 0), 3)
                        cv2.circle(im, (box[l, 3], box[l, 4]), 3, (255, 0, 0), 3)

                    cv2.imwrite('./trainset_check/' + str(self.show_id) + '.jpg', im)

            img = img / 255.
            new_image = torch.from_numpy(img).permute(2, 0, 1).float()
            try:
                new_boxes = torch.from_numpy(boxes).view(1, -1)
            except:
                new_boxes = torch.from_numpy(boxes.reshape(1, -1))

            return img_path, new_image, new_boxes
        else:
            print("label_path is empty!!!!!!!")
            return [], [], []

    def collate_fn(self, batch):

        paths, imgs, labels = list(zip(*batch))
        imgs = torch.stack([img for img in imgs])
        return paths, imgs, labels

    def __len__(self):
        # #该函数定义dataloader输出长度，若长度大于样本数，则重复之前输出
        return len(self.img_paths)


class DETECT_IMG(Dataset):
    def __init__(self, img_path, img_size):
        self.img_path = img_path
        self.img_size = img_size

    def __getitem__(self, index):

        img_path = self.img_path
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        if img.shape[0] > img.shape[1]:
            trans = False
        else:
            trans = False

        img = torch.from_numpy(img)
        boxes = []

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        new_img, _ = resize_image(img, boxes, size=self.img_size)

        return img_path, new_img

    def __len__(self):
        return 1


def get_val(path,
            train_path,
            val_path,
            val_index,
            epochs,
            train_img_size,
            batch_size,
            num_workers,
            aug_paras,
            data_aug):

    generate_val(path=path,
                 train_path=train_path,
                 val_path=val_path,
                 val_index=val_index,
                 epochs=epochs)

    train_dataset = YOLO4Dataset(train_path, img_size=train_img_size, aug_paras=aug_paras, data_aug=data_aug)
    val_dataset = YOLO4Dataset(val_path, img_size=train_img_size, aug_paras=aug_paras, data_aug=data_aug)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn
                                  )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                collate_fn=val_dataset.collate_fn
                                )

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    imgg = cv2.imread('./cor_4.jpg')
    test = False
    b1 = np.loadtxt('./label_draft.txt').reshape(-1, 5)
    b2 = np.loadtxt('./label_draft2.txt').reshape(-1, 5)

    _, t_b1 = resize_image(imgg, b1, size=608)
    _, t_b2 = resize_image(imgg, b2, size=608)

    t_b = (t_b1, t_b2)

    anchors = torch.tensor([[[93,93],
                             [95,96],
                             [99,98]],
                            [[102,101],
                             [106,106],
                             [108,108]],
                            [[113,112],
                             [119,119],
                             [131,130]]])
    stride = torch.tensor([8., 16., 32.])

    # print(t_b1.dtype)

    gt = generate_groundtruth(t_b,
                              input_size=608,
                              anchors=anchors,
                              stride=stride,
                              num_classes=20)

    # print(gt[0].size())
    # print(gt[1].size())
    # print(gt[2].size())





