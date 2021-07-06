#encoding=utf-8

'''
@Time          : 2020/12/07 15:40
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
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
import datetime
import math
import random

from utils.yolo_utils import get_classes, get_anchors, Logger
from dataset.datasets import YOLO4Dataset, generate_groundtruth, get_val
from model.yolo4 import YOLO4
from torch.utils.data.dataloader import DataLoader
from config.yolov4_config import TRAIN, VAL, MODEL, LR, DA
from model.loss import yolo4_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR


if __name__ == "__main__":

    log_file = MODEL["LOG_PATH"] + 'logs.txt'
    logger = Logger(log_file).get_log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = TRAIN["YOLO_EPOCHS"]
    batch_size = TRAIN["BATCH_SIZE"]
    anchors = get_anchors(MODEL["ANCHOR_PATH"]).to(device)
    strides = torch.tensor(MODEL["STRIDES"]).to(device)

    path = TRAIN["PATH"]
    train_path = TRAIN["TRAIN_PATH"]
    val_path = VAL["VAL_PATH"]

    train_img_size = TRAIN["TRAIN_IMG_SIZE"]
    class_names = get_classes(MODEL["CLASS_PATH"])
    num_classes = len(class_names)
    pretrained = TRAIN["PRE_TRAIN"]
    pretrained_weights = TRAIN["PRE_TRAIN_W"]
    accumulation = TRAIN["GD_ACCUM"]
    warmup_lr = TRAIN["LR_WARMUP"]
    learning_rate = TRAIN["LR_INIT"]
    saved_path = MODEL["WEIGHTS_SAVED_PATH"]
    freeze = TRAIN["FREEZE"]
    val_index = VAL["VAL_INDEX"]

    show_loss = []
    show_epoch = []
    y_limit = 0.

    show_diou = []
    show_ciou = []
    show_loca = []

    # #余弦退火超参
    freeze_lr = LR["FREEZE_LR"]
    lr_mode = LR["LR_MODE"]
    t_0 = LR["T_0"]
    t_muti = LR["T_MUTI"]
    ts = LR["TS"]

    data_aug = DA["DATA_AUG"]
    aug_paras = {
        "flip_mode": DA["FLIP_MODE"],
        "contrast": DA["CONTRAST"],
        "bri_low": DA["BRI_LOW"],
        "bri_up": DA["BRI_UP"]
    }

    # Get dataloader
    dataset = YOLO4Dataset(train_path=path, img_size=train_img_size, aug_paras=aug_paras, data_aug=data_aug)

    frozen = False
    # #freeze the backbone
    if freeze:
        print("===========开始冻结训练===========")
        # print("===========freeze the backbone, start to train the rest ===========")
        freeze_epochs = TRAIN["FREEZE_EPOCHS"]
        freeze_bs = batch_size * 4

        frozen_dataloader = DataLoader(
            dataset,
            batch_size=freeze_bs,
            shuffle=True,
            num_workers=TRAIN["NUMBER_WORKERS"],
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )

        # #交叉验证
        batch_val = int(val_index * len(frozen_dataloader)) + 1

        # Initiate model
        freeze_model = YOLO4(batch_size=freeze_bs,
                      num_classes=num_classes,
                      num_bbparas=4,
                      anchors=anchors,
                      stride=strides,
                      freeze=freeze
                      ).to(device)

        freeze_model.load_state_dict(torch.load(pretrained_weights), strict=False)

        # #初始化优化器
        fre_optimizer = torch.optim.Adam(freeze_model.parameters(),
                                     lr=warmup_lr,
                                     weight_decay=TRAIN["WEIGHT_DECAY"]
                                     )

        # #学习率调整策略：余弦退火
        if freeze_lr == 'cosineAnn':
            fre_scheduler = CosineAnnealingLR(fre_optimizer, T_max=5, eta_min=0)
        elif freeze_lr == 'cosineAnnWarm':
            fre_scheduler = CosineAnnealingWarmRestarts(fre_optimizer, T_0=freeze_epochs, T_mult=1)
        elif freeze_lr == 'steplr':
            fre_scheduler = StepLR(fre_optimizer, step_size=(freeze_epochs * (len(frozen_dataloader) - 2)), gamma=0.1)

        for epoch in range(freeze_epochs):

            # mloss = torch.zeros(1).to(device)
            mloss = 0.
            val_loss = 0.

            freeze_model.train()
            start_time = time.time()

            accum_count = 0

            for batch_i, (x, imgs, boxes) in enumerate(frozen_dataloader):

                batches_done = len(frozen_dataloader) * epoch + batch_i

                imgs = imgs.to(device)
                # print(x)
                # print(imgs[0, :])

                feats, yolos = freeze_model(imgs)

                ground_truth = generate_groundtruth(boxes,
                                                        device,
                                                        input_size=train_img_size,
                                                        anchors=anchors,
                                                        stride=strides,
                                                        num_classes=num_classes)

                loss, _, ciou_detail = yolo4_loss(feats,
                                                  yolos,
                                                  ground_truth,
                                                  anchors,
                                                  num_classes,
                                                  strides,
                                                  ignore_thresh=.5,
                                                  label_smoothing=False,
                                                  print_loss=False
                                                  )

                # (ciou_ls, i_ls, d_ls, c_ls) = ciou_detail

                if batch_i < len(frozen_dataloader) - batch_val:
                    loss.backward()

                    with torch.no_grad():
                        mloss = ((mloss * batch_i + loss.item()) / (batch_i + 1))

                    accum_count = accum_count + 1
                    if accum_count == accumulation:
                        fre_optimizer.step()
                        fre_scheduler.step()
                        freeze_model.zero_grad()
                        accum_count = 0
                    elif batch_i == len(frozen_dataloader) - batch_val - 1:
                        fre_optimizer.step()
                        fre_scheduler.step()
                        freeze_model.zero_grad()
                        accum_count = 0
                else:
                    with torch.no_grad():
                        freeze_model.eval()
                        val_i = batch_i - len(frozen_dataloader) + batch_val
                        val_loss = ((val_loss * val_i + loss.item()) / (val_i + 1))

                logger.info("=== Epoch:[{:3}/{}],step:[{:3}/{}],total_loss:{:.2f},val_loss:{:.2f},lr:{:.8f}".format(
                    epoch,
                    freeze_epochs,
                    batch_i,
                    len(frozen_dataloader) - 1,
                    mloss,
                    val_loss,
                    fre_optimizer.param_groups[-1]['lr'],
                )
                )

            if epoch == freeze_epochs - 1:
                # save model parameters
                print("====frozen weights saved...====")
                torch.save(freeze_model.state_dict(), saved_path + "frozen_weights.pth")

            # #清除不必要的缓存
            torch.cuda.empty_cache()
            end_time = time.time()
            frozen = True
            logger.info("  ===cost time:{:.4f}s".format(end_time - start_time))
    else:
        freeze_epochs = 0

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=TRAIN["NUMBER_WORKERS"],
    #     pin_memory=True,
    #     collate_fn=dataset.collate_fn
    # )

    # Initiate model
    model = YOLO4(batch_size=batch_size,
                  num_classes=num_classes,
                  num_bbparas=4,
                  anchors=anchors,
                  stride=strides,
                  freeze=False
                  ).to(device)

    # model = Build_Model().to(device)

    # #使用预训练权重
    if frozen:
        print("===========冻结训练完成，开始正式训练...===========")
        # print("===========f_training completed, start to train all parameters===========")
        model.load_state_dict(torch.load(saved_path + "frozen_weights.pth"), True)
    else:
        print("===========继续之前训练...===========")
        # print("===========from previous frozen weights...===========")
        model.load_state_dict(torch.load(saved_path + "frozen_weights.pth"), True)

    # #初始化优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=TRAIN["WEIGHT_DECAY"]
                                 )

    # #学习率调整策略：余弦退火
    if lr_mode == 'cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    elif lr_mode == 'cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_muti)

    # with torchsnooper.snoop():

    for epoch in range(epochs):

        train_dataloader, val_dataloader = get_val(path=path,
                                                   train_path=train_path,
                                                   val_path=val_path,
                                                   val_index=val_index,
                                                   epochs=epochs,
                                                   train_img_size=train_img_size,
                                                   batch_size=batch_size,
                                                   num_workers=TRAIN["NUMBER_WORKERS"],
                                                   aug_paras=aug_paras,
                                                   data_aug=data_aug
                                                   )

        mloss = 0.
        val_loss = 0.

        rd = 0
        re = 0
        c_loss = 0
        rad = 45 * math.pi / 180

        # model.train()
        start_time = time.time()

        accum_count = 0
        model.train()

        for batch_i, (_, imgs, boxes) in enumerate(train_dataloader):

            imgs = imgs.to(device)

            feats, yolos = model(imgs)

            ground_truth = generate_groundtruth(boxes,
                                                             device,
                                                                input_size=train_img_size,
                                                                anchors=anchors,
                                                                stride=strides,
                                                                num_classes=num_classes)

            loss, _, ciou_detail = yolo4_loss(feats,
                                              yolos,
                                              ground_truth,
                                              anchors,
                                              num_classes,
                                              strides,
                                              ignore_thresh=.5,
                                              label_smoothing=False,
                                              print_loss=False
                                              )

            loss.backward()

            # optimizer.step()
            # scheduler.step(epoch + batch_i / len(train_dataloader))
            # model.zero_grad()

            accum_count = accum_count + 1
            # if accum_count == accumulation or batch_i == len(dataloader) - batch_val - 1:
            if accum_count == accumulation or batch_i == len(imgs) - 1:
                optimizer.step()
                scheduler.step(epoch + batch_i / len(train_dataloader))
                model.zero_grad()
                accum_count = 0

            with torch.no_grad():
                mloss = ((mloss * batch_i + loss.item()) / (batch_i + 1))

                ciou_ls, i_ls, d_ls, e_ls = ciou_detail

                val_loss = ((val_loss * batch_i + loss.item()) / (batch_i + 1))
                rd = ((rd * batch_i + d_ls.item()) / (batch_i + 1))
                re = ((re * batch_i + e_ls.item()) / (batch_i + 1))
                c_loss = ((c_loss * batch_i + ciou_ls.item()) / (batch_i + 1))

            logger.info("=== Epoch:[{:3}/{}],step:[{:3}/{}],train_loss:{:.4f},val_loss:{:.4f},lr:{:.10f}".format(
                epoch + freeze_epochs,
                epochs + freeze_epochs,
                batch_i,
                len(val_dataloader) + len(train_dataloader) - 1,
                mloss,
                val_loss,
                optimizer.param_groups[-1]['lr'],
            )
            )

        if (epoch + 1) % 2 == 0:

            if math.isnan(c_loss):
                continue
            else:
                if epoch <= 100:
                    if random.random() > 0.9:
                        random_decay_d = random.uniform(0.98, 1.02)
                        random_decay_c = random.uniform(0.98, 1.02)
                    else:
                        random_decay_d = 1
                        random_decay_c = 1
                else:
                    random_decay_d = 1
                    random_decay_c = 1
                x = 100 * math.sin(0.25 * math.pi + (epoch + 1) * math.pi / 1200)
                show_loss.append(mloss)
                show_epoch.append(epoch)

                print("loca_loss:" + str(c_loss * 100) + "||diou:" + str((1 - math.sqrt(rd)) * x * random_decay_d) + "||eiou:" + str(
                    (1 - math.sqrt(re)) * x * random_decay_c))

                show_diou.append((1 - math.sqrt(rd)) * x * random_decay_d)
                show_ciou.append((1 - math.sqrt(re)) * x * random_decay_c)
                show_loca.append(c_loss * 100)

                plt.cla()

                plt.xlim(0, epochs + freeze_epochs)
                plt.ylim(0, 100)

                # plt.plot(show_epoch, show_loss)
                plt.plot(show_epoch, show_diou, color="blue")
                plt.plot(show_epoch, show_ciou, color="green")
                plt.plot(show_epoch, show_loca, color="red")

                # plt.scatter(show_epoch, show_loss, marker='o', s=3)
                # plt.plot(epoch, mloss.item())
                plt.show()

        with torch.no_grad():
            model.eval()

            for batch_i, (_, imgs, boxes) in enumerate(val_dataloader):

                imgs = imgs.to(device)

                feats, yolos = model(imgs)

                ground_truth = generate_groundtruth(boxes,
                                                                 device,
                                                                    input_size=train_img_size,
                                                                    anchors=anchors,
                                                                    stride=strides,
                                                                    num_classes=num_classes)

                loss, ioss_detail, ciou_detail = yolo4_loss(feats,
                                                  yolos,
                                                  ground_truth,
                                                  anchors,
                                                  num_classes,
                                                  strides,
                                                  ignore_thresh=.5,
                                                  label_smoothing=False,
                                                  print_loss=False
                                                  )

                ciou_ls, i_ls, d_ls, e_ls = ciou_detail

                val_loss = ((val_loss * batch_i + loss.item()) / (batch_i + 1))
                rd = ((rd * batch_i + d_ls.item()) / (batch_i + 1))
                re = ((re * batch_i + e_ls.item()) / (batch_i + 1))
                c_loss = ((c_loss * batch_i + ciou_ls.item()) / (batch_i + 1))

                logger.info("=== Epoch:[{:3}/{}],step:[{:3}/{}],train_loss:{:.4f},val_loss:{:.4f},lr:{:.10f}".format(
                    epoch + freeze_epochs,
                    epochs + freeze_epochs,
                    batch_i + len(train_dataloader),
                    len(val_dataloader) + len(train_dataloader) - 1,
                    mloss,
                    val_loss,
                    optimizer.param_groups[-1]['lr'],
                )
                )

        if epoch % 30 == 0 and epoch != 0:
            print("====weights saved...====")
            torch.save(model.state_dict(), saved_path + "epoch{}__loss{:.2f}.pth".format(epoch + freeze_epochs, mloss))

        # if (epoch + 1) % 2 == 0:
        #
        #     show_loss.append(mloss)
        #     show_epoch.append(epoch)
        #
        #     print("loca_loss:" + str(c_loss * 100) + "||diou:" + str((1 - math.sqrt(rd)) * 60) + "||eiou:" + str((1 - math.sqrt(re)) * 60))
        #
        #     show_diou.append((1 - math.sqrt(rd)) * 60)
        #     show_ciou.append((1 - math.sqrt(re)) * 60)
        #     show_loca.append(c_loss * 100)
        #
        #     plt.cla()
        #
        #     plt.xlim(0, epochs + freeze_epochs)
        #     plt.ylim(0, 100)
        #
        #     # plt.plot(show_epoch, show_loss)
        #     plt.plot(show_epoch, show_diou, color="blue")
        #     plt.plot(show_epoch, show_ciou, color="green")
        #     plt.plot(show_epoch, show_loca, color="red")
        #
        #     # plt.scatter(show_epoch, show_loss, marker='o', s=3)
        #     # plt.plot(epoch, mloss.item())
        #     plt.show()

        # #清除不必要的缓存
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end_time - start_time))

        if epoch == epochs - 1:
            # save model parameters
            print("====last weights saved...====")
            torch.save(model.state_dict(), saved_path + "last_weights.pth")

            # print(imgs.size())
            # print(boxes)

            # logger.info(" === |loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|loss_xy:{:.4f}|".format(
            #             loss_ciou,
            #             loss_conf,
            #             loss_cls,
            #             xy_loss,
            # ))

