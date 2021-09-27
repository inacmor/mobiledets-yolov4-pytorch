import numpy as np
import torch
import time
import cv2
import apex

from model.net.csp_mobiledets import CSPMobileDets
from model.net.mobiledet_gpu import MobileDetGPU
from model.net.backbone import CSPDarknet53

if __name__ == '__main__':

    # img = np.random.randint(low=0, high=255, size=(720, 1280, 3))
    # # img = cv2.imencode('.jpg', img.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # # img = img.tobytes()
    # cv2.imwrite('./666.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])

    # img = cv2.imread('./pics/IMG_5671.JPG')
    # img_encode = np.getbuffer(img)
    # print(img.tobytes())

    with torch.no_grad():
        device = torch.device("cpu")

        timeall = 0
        img = torch.rand(size=(1, 3, 416, 416)).to(device)

        model = CSPMobileDets().to(device)
        model.eval()

        # c1, c2, c3 = model(img)
        # print(c1.size())
        # print(c2.size())
        # print(c3.size())

        for i in range(50):
            start = time.time()

            c1, c2, c3 = model(img)

            print(c1.size())
            print(c2.size())
            print(c3.size())

            end = time.time()
            if i == 0:
                continue
            else:
                timeall = timeall + end - start

            print(end - start)

        print("avg time: ", timeall * 1000 / 50, " ms")
