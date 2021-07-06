import os
import torch
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def background_mix(img, box, clips, index, batchsize, do=False):

    backup = img.copy()

    size = img.shape[0]

    # #抽取切片
    top_limit = np.max(box[1])
    lef_limit = np.max(box[0])
    bot_limit = np.max(box[3])
    rig_limit = np.max(box[2])

    top = backup[0:random.randint(0, top_limit), :, :]
    left = backup[:, 0:random.randint(0, lef_limit), :]
    bottom = backup[random.randint(bot_limit, size):size, :, :]
    right = backup[:, random.randint(rig_limit, size):size, :]

    top_shape = (random.randint(0, top_limit), random.randint(size / 2, size))
    left_shape = (random.randint(size / 2, size), random.randint(0, lef_limit))
    bot_shape = (random.randint(0, size - bot_limit), random.randint(size / 2, size))
    rig_shape = (random.randint(size / 2, size), random.randint(0, size - rig_limit),)

    if index < batchsize - 1:
        clips.append((top, left, bottom, right))
        if index == 0:
            return img, clips
        else:
            top_c, left_c, bottom_c, right_c = clips[index - 1]

    else:
        top_c = np.hstack((cv2.resize(clips[0][0], top_shape),
                           cv2.resize(clips[1][0], top_shape),
                           cv2.resize(clips[2][0], top_shape)))
        left_c = np.vstack((cv2.resize(clips[0][1], left_shape),
                            cv2.resize(clips[1][1], left_shape),
                            cv2.resize(clips[2][1], left_shape)))
        bottom_c = np.hstack((cv2.resize(clips[0][2], bot_shape),
                              cv2.resize(clips[1][2], bot_shape),
                              cv2.resize(clips[2][2], bot_shape)))
        right_c = np.vstack((cv2.resize(clips[0][3], rig_shape),
                             cv2.resize(clips[1][3], rig_shape),
                             cv2.resize(clips[2][3], rig_shape)))

    top_c = cv2.resize(top_c, (top_shape[1], top_shape[0]))
    left_c = cv2.resize(left_c, (left_shape[1], left_shape[0]))
    bottom_c = cv2.resize(bottom_c, (bot_shape[1], bot_shape[0]))
    right_c = cv2.resize(right_c, (rig_shape[1], rig_shape[0]))

    top_startx = random.randint(0, size - top_shape[1])
    top_starty = random.randint(0, top_limit - top_shape[0])
    left_startx = random.randint(0, lef_limit - left_shape[1])
    left_starty = random.randint(0, size - left_shape[0])
    bottom_startx = random.randint(0, size - bot_shape[1])
    bottom_starty = random.randint(bot_limit, size - bot_shape[0])
    right_startx = random.randint(rig_limit, size - rig_shape[1])
    right_starty = random.randint(0, size - rig_shape[0])

    img[top_starty:top_starty + top_shape[0], top_startx:top_startx + top_shape[1], :] = top_c
    img[left_starty:left_starty + left_shape[0], left_startx:left_startx + left_shape[1], :] = left_c

    img[bottom_starty:bottom_starty + bot_shape[0], bottom_startx:bottom_startx + bot_shape[1], :] = bottom_c
    img[right_starty:right_starty + rig_shape[0], right_startx:right_startx + rig_shape[1], :] = right_c

    return img, clips


if __name__=='__main__':

    # #test
    clips = []
    boxes = np.array([[967, 601, 2033, 1673],
                      [1039, 560, 2117, 1639],
                      [1057, 778, 1927, 1642],
                      [1046, 631, 2049, 1619]])
    for i in range(4):

        box = boxes[i, ...]
        img = cv2.imread('./data/bg_mix/IMG_561' + str(5+i) + '.JPG')

        new_img, clips = background_mix(img, box, clips, i, 4)
        print("done...")

        cv2.imwrite('./data/bg_mix/' + str(i) + '.jpg', new_img)


    # # #stack
    # path = './data/bg_mix/'
    #
    # # img1 = cv2.imread(path+'0.JPG')
    # # img2 = cv2.imread(path + '1.JPG')
    # # img3 = cv2.imread(path + '2.JPG')
    # # img4 = cv2.imread(path + '3.JPG')
    # img1 = cv2.imread(path + 'IMG_5615.JPG')
    # img2 = cv2.imread(path + 'IMG_5616.JPG')
    # img3 = cv2.imread(path + 'IMG_5617.JPG')
    # img4 = cv2.imread(path + 'IMG_5618.JPG')
    #
    # ori_stack = np.hstack((img1, img2, img3, img4))
    #
    # cv2.imwrite(path+'ori_stack.jpg', ori_stack)

    # rad = 45 * math.pi / 180
    # print(math.sin(0.25 * math.pi))














