# mobiledets-yolov4-pytorch
This is an object detecting model based on yolov4.

I replace the backbone to mobiledets:https://arxiv.org/abs/2004.14525.

The mobiledets model codes are mainly from https://github.com/novice03/mobiledet-pytorch.

---Train---
- step1: put your trainset and labels(xml) in data/Imgs and data/Annotations.
- step2: put your pretrained .pth in weights/pretrained.
- step3: run ready_for_training.py.
- step4: run kmeans.py.
- step5: run train.py.

---Inference---
- step1: put your testset in data/Test_I.
- step2: run detect.py.
You can find the results in./devs.

---requirement---
- pytorch1.4.0.
- timm.
- tucker_conv.
- mish_cuda
