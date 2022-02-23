# mobiledets-yolov4-pytorch
# reach me on 862582838@qq.com.
This is an object detecting model based on yolov4.

I replace the backbone to mobiledets:https://arxiv.org/abs/2004.14525.
(updated 20210927: add res_block and csp block to backbone. So now we can call it CSPmobiledets-yolov4 I think.)

---Train---
- step1: put your trainset and labels(xml) in data/Imgs and data/Annotations.
- step2: put your pretrained .pth in weights/pretrained.
- step3: run 'python ready.py'.
- step4: run 'python kmeans.py'.
- step5: run 'python train.py'.

---Inference---
- step1: put your testset in devs.
- (if pictures)step2: run 'python pic_detect.py';You can find results in devs/detect_results.
- (if videos)step2: run 'python vision_test.py'.

---requirement---
- pytorch1.7.0.(https://pytorch.org/get-started/previous-versions/)
