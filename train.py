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
import matplotlib.pyplot as plt
import numpy as np
from apex import amp

import os
import time
import sys

from utils.yolo_utils import get_classes, get_anchors, Logger
from dataset.datasets import YOLO4Dataset, generate_groundtruth, get_val
from model.yolo4 import YOLO4
from config.yolov4_config import TRAIN, VAL, MODEL, LR, DA
from model.loss import yolo4_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR


def get_inputs(freeze, frozen=False, apex_speedup=False):

    if freeze:
        epochs = TRAIN["FREEZE_EPOCHS"]
        batch_size = TRAIN["BATCH_SIZE"] * 4
        lr = TRAIN["LR_WARMUP"]

    else:
        epochs = TRAIN["YOLO_EPOCHS"]
        batch_size = TRAIN["BATCH_SIZE"]
        lr = TRAIN["LR_INIT"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchors = get_anchors(MODEL["ANCHOR_PATH"]).to(device)
    strides = torch.tensor(MODEL["STRIDES"]).to(device)

    path = TRAIN["PATH"]
    train_path = TRAIN["TRAIN_PATH"]
    val_path = VAL["VAL_PATH"]

    train_img_size = TRAIN["TRAIN_IMG_SIZE"]
    class_names = get_classes(MODEL["CLASS_PATH"])
    num_classes = len(class_names)

    accumulation = TRAIN["GD_ACCUM"]
    saved_epoch = TRAIN["SAVE_EPOCH"]
    saved_path = MODEL["WEIGHTS_SAVED_PATH"]
    if not frozen:
        pretrained_weights = TRAIN["PRE_TRAIN_W"]
    else:
        pretrained_weights = saved_path + "frozen_weights.pth"

    val_index = VAL["VAL_INDEX"]

    # #Cosine annealing
    t_0 = LR["T_0"]
    t_muti = LR["T_MUTI"]

    data_aug = DA["DATA_AUG"]
    aug_paras = {
        "flip_mode": DA["FLIP_MODE"],
        "contrast": DA["CONTRAST"],
        "bri_low": DA["BRI_LOW"],
        "bri_up": DA["BRI_UP"]
    }

    # #initiate model
    model = YOLO4(batch_size=batch_size,
                  num_classes=num_classes,
                  num_bbparas=4,
                  anchors=anchors,
                  stride=strides,
                  freeze=freeze
                  ).to(device)
    model.load_state_dict(torch.load(pretrained_weights), strict=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=TRAIN["WEIGHT_DECAY"]
                                 )

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

    if freeze:
        scheduler = StepLR(optimizer, step_size=(epochs // 2), gamma=0.1)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_muti)

    if apex_speedup:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    paras = [device,
             epochs,
             saved_epoch,
             train_img_size,
             anchors,
             num_classes,
             strides,
             accumulation,
             saved_path]

    return paras, [train_dataloader, val_dataloader], model, [scheduler, optimizer]


def train(freeze, paras, loaders, model, backs, logger, draw=True, apex_speedup=False):

    # paras, loaders, model, backs = get_inputs()

    device, epochs, saved_epoch, train_img_size, anchors, num_classes, strides, accumulation, saved_path = paras
    train_dataloader, val_dataloader = loaders
    scheduler, optimizer = backs

    show_loss = []
    show_epoch = []

    for epoch in range(epochs):

        mloss = 0.
        val_loss = 0.

        # model.train()
        start_time = time.time()

        accum_count = 0
        model.train()

        for batch_i, (_, imgs, boxes) in enumerate(train_dataloader):

            imgs = imgs.to(device)

            feats, yolos = model(imgs)

            ground_truth = generate_groundtruth(boxes, device, train_img_size, anchors, strides, num_classes)

            loss = yolo4_loss(feats,
                              yolos,
                              ground_truth,
                              ignore_thresh=.5,
                              )

            if apex_speedup:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            accum_count = accum_count + 1
            # if accum_count == accumulation or batch_i == len(dataloader) - batch_val - 1:
            if accum_count == accumulation or batch_i == len(imgs) - 1:
                optimizer.step()
                scheduler.step(epoch + batch_i / len(train_dataloader))
                model.zero_grad()
                accum_count = 0

            with torch.no_grad():
                mloss = ((mloss * batch_i + loss.item()) / (batch_i + 1))

            logger.info("=== Epoch:[{:3}/{}],step:[{:3}/{}],train_loss:{:.2f},val_loss:{:.2f},lr:{:.10f}".format(
                epoch,
                epochs,
                batch_i,
                len(val_dataloader) + len(train_dataloader) - 1,
                mloss,
                val_loss,
                optimizer.param_groups[-1]['lr'],
            )
            )

        with torch.no_grad():
            model.eval()

            for batch_i, (_, imgs, boxes) in enumerate(val_dataloader):

                imgs = imgs.to(device)

                feats, yolos = model(imgs)

                ground_truth = generate_groundtruth(boxes, device, train_img_size, anchors, strides, num_classes)

                loss = yolo4_loss(feats,
                                  yolos,
                                  ground_truth,
                                  ignore_thresh=.5
                                  )

                val_loss = ((val_loss * batch_i + loss.item()) / (batch_i + 1))

                logger.info("=== Epoch:[{:3}/{}],step:[{:3}/{}],train_loss:{:.4f},val_loss:{:.4f},lr:{:.10f}".format(
                    epoch,
                    epochs,
                    batch_i + len(train_dataloader),
                    len(val_dataloader) + len(train_dataloader) - 1,
                    mloss,
                    val_loss,
                    optimizer.param_groups[-1]['lr'],
                )
                )

        if not freeze:

            freeze_epochs = TRAIN["FREEZE_EPOCHS"]

            if epoch % 30 == 0 and epoch != 0:
                print("====weights saved...====")
                torch.save(model.state_dict(), saved_path + "epoch{}__loss{:.2f}.pth".format(epoch + freeze_epochs, mloss))

        if draw:
            if (epoch + 1) % 3 == 0:
                show_loss.append(mloss)
                show_epoch.append(epoch)

                plt.cla()

                plt.plot(show_epoch, show_loss)
                plt.show()

        # #release unnessasery cache
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end_time - start_time))

        if epoch == epochs - 1:
            if freeze:
                print("====frozen weights saved...====")
                torch.save(model.state_dict(), saved_path + "frozen_weights.pth")
            else:
                print("====last weights saved...====")
                torch.save(model.state_dict(), saved_path + "last_weights.pth")


if __name__ == "__main__":

    freeze = TRAIN["FREEZE"]
    log_file = MODEL["LOG_PATH"] + 'logs.txt'
    logger = Logger(log_file).get_log()
    paras, loaders, model, backs = get_inputs(freeze, frozen=False)

    try:
        if freeze:
            print("================start================")
            print("================warmming up================")
            train(freeze, paras, loaders, model, backs, logger, draw=False)
            print("================formal training================")
            paras, loaders, model, backs = get_inputs(False, frozen=True)
            train(False, paras, loaders, model, backs, logger, draw=True)
        else:
            print("================start================")
            train(freeze, paras, loaders, model, backs, logger, draw=True)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


