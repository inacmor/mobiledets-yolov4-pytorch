
# coding=utf-8

# train
TRAIN = {
    "PATH": "./data_train.txt",
    "TRAIN_PATH": "./validation/train.txt",
    "FREEZE": True,
    "TRAIN_IMG_SIZE": 608,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": False,
    "IOU_THRESHOLD_LOSS": 0.5,
    "FREEZE_EPOCHS": 20,
    "YOLO_EPOCHS": 90,
    "NUMBER_WORKERS": 0,
    "MOMENTUM": 0.937,
    "WEIGHT_DECAY": 0.0003,
    "LR_WARMUP": 0.001,
    "LR_INIT": 0.0001,
    "LR_END": 1e-6,
    "PRE_TRAIN": True,
    "PRE_TRAIN_W": './weights/pretrained/yolov4.pth',
    "GD_ACCUM": 1,
    "SAVE_EPOCH": 10
}

# val
VAL = {
    "VAL_INDEX": 0.05,
    "VAL_PATH": './validation/val.txt',
    "TEST_IMG_SIZE": 608,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.5,
    "NMS_THRESH": 0.5,
}

# model
MODEL = {
    "ANCHOR_PATH": './data/yolo_anchors.txt',
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
    "CLASS_PATH": './data/classes.txt',
    "LOG_PATH": './logs/',
    "WEIGHTS_SAVED_PATH": './weights/output/'
}

# #learning_rate
LR = {
    "FREEZE_LR": 'steplr',
    "LR_MODE": 'cosineAnnWarm',
    "T_0": 30,
    "T_MUTI": 2,
    "TS": 1
}

DA = {
    "DATA_AUG": True,
    "FLIP_MODE": 'hori',
    "CONTRAST": 1.2,
    "BRI_LOW": -40,
    "BRI_UP": 30
}

DETECT = {
    "BATCH_SIZE": 1,
    "DETECT_SIZE": 608,
    "ANCHOR_PATH": './data/yolo_anchors.txt',
    "STRIDES": [8, 16, 32],
    "CLASS_PATH": './data/classes.txt',
    "WEIGHT_PATH": './weights/output/last_weights.pth',
    "IMG_PATH": './devs/left24.jpg',
    # "IMG_PATH": './tv/Test_I/JJ_none_none_none_train_p_day_20210417_11.jpg',
    "CONF_THRES": 0.5,
    "NMS_THRES": 0.5,
}

