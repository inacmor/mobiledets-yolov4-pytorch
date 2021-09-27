#encoding=utf-8

'''
@Time          : 2021/04/23 19:19
@Author        : Inacmor
@File          : vision_test.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import torch
import os
import time
import cv2

from utils.detect_supports import initialize_model, predict

if __name__ == "__main__":

    model, device = initialize_model()

    cap = cv2.VideoCapture(0)

    cap.set(3, 1080)
    cap.set(4, 720)

    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()

        n_frame = predict(frame, model, device)

        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()