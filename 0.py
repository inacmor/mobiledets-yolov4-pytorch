import torch
import numpy as np
import cv2
import time
from model.yolo4 import YOLO4
from model.net.backbone import Darknet53


img = cv2.imread('./data/bg_mix/IMG_5618.JPG')
cv2.rectangle(img, [1046, 631],[2049, 1619], 2)
cv2.imshow('0', img)
cv2.waitKey(0)
cv2.destroyAllWindows()