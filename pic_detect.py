#encoding=utf-8

'''
@Time          : 2021/05/10 22:23
@Author        : Inacmor
@File          : pic_detect.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import os
import cv2

from utils.detect_supports import predict, initialize_model


if __name__ == "__main__":

    model, device = initialize_model()

    list = os.listdir('./pics/')

    for l, n in enumerate(list):

        if n == 'detect_results':
            continue

        path = './pics/' + list[l]
        img = cv2.imread(path)

        detected_img = predict(img, model, device)

        cv2.imwrite('./pics/detect_results/' + n, detected_img)




