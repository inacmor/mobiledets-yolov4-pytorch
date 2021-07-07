import cv2
import numpy as np
import random


def flip(img, box, mode):
    s = img.shape[0]

    if mode == 'hori' or 'vert':
        if random.random() > 0.5:
            if mode == 'hori':
                img = cv2.flip(img, 1)
                box[:, 1] = s - box[:, 1]
                box[:, 3] = s - box[:, 3]
                box = box[:, [0, 3, 2, 1, 4]]
            else:
                img = cv2.flip(img, 0)
                box[:, 2] = s - box[:, 2]
                box[:, 4] = s - box[:, 4]
                box = box[:, [0, 1, 4, 3, 2]]

    if mode == 'mix':
        if random.random() > 0.25:
            flip_id = [1, 0, -1]
            i = random.choice(flip_id)

            if i == 1:
                img = cv2.flip(img, 1)
                box[:, 1] = s - box[:, 1]
                box[:, 3] = s - box[:, 3]
                box = box[:, [0, 3, 2, 1, 4]]
            elif i == 0:
                img = cv2.flip(img, 0)
                box[:, 2] = s - box[:, 2]
                box[:, 4] = s - box[:, 4]
                box = box[:, [0, 1, 4, 3, 2]]
            else:
                img = cv2.flip(img, -1)
                box[:, 1] = s - box[:, 1]
                box[:, 3] = s - box[:, 3]
                box = box[:, [0, 3, 2, 1, 4]]
                box[:, 2] = s - box[:, 2]
                box[:, 4] = s - box[:, 4]
                box = box[:, [0, 1, 4, 3, 2]]

    return img, box


def rotate(img, box):

    s = img.shape[0]

    if random.random() > 0.33:

        times = [1, 3]
        time_id = random.choice(times)

        if time_id == 1:
            img = np.rot90(img)
            box = box[:, [0, 2, 3, 4, 1]]
            box[:, 2] = s - box[:, 2]
            box[:, 4] = s - box[:, 4]
        elif time_id == 3:
            img = np.rot90(np.rot90(np.rot90(img)))
            box = box[:, [0, 2, 3, 4, 1]]
            box[:, 1] = s - box[:, 1]
            box[:, 3] = s - box[:, 3]

    return img, box


def convert(img, contrast, bri_low, bri_up):

    contrast = random.uniform(1, contrast)
    bright = random.uniform(bri_low, bri_up)

    img = contrast * img + bright

    img = np.minimum(np.maximum(img, 0), 255)

    return img


def background_cutmix(img, box, clips, index, batchsize):
    backup = img.copy()

    size = img.shape[0]

    top_limit = np.max(box[:, 1])
    lef_limit = np.max(box[:, 0])
    bot_limit = np.max(box[:, 3])
    rig_limit = np.max(box[:, 2])

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
