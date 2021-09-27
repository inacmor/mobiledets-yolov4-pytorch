#encoding=utf-8

'''
@Time          : 2021/09/20 10:40
@Author        : Inacmor
@File          : ready.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import xml.etree.ElementTree as ET
from os import getcwd, listdir
from utils.yolo_utils import get_classes

xmlfilepath=r'./data/Annotations/'
classfilepath=r'./data/classes.txt'


def convert_annotation(anno_path='./data/Annotations/', saved_path='data_train.txt'):

    wd = getcwd()

    classes = get_classes(classfilepath)

    xml_files = listdir(anno_path)

    w_id = 0

    with open(saved_path, 'a+') as sf:

        sf.truncate(0)

        for f in xml_files:
            in_file = open((anno_path + f), 'rb')
            tree = ET.parse(in_file)
            root = tree.getroot()

            name = root.find('filename').text
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                # print(xmlbox.find('xmin').text)
                # print(xmlbox.find('ymin').text)
                b = (int(float(xmlbox.find('xmin').text)),
                     int(float(xmlbox.find('ymin').text)),
                     int(float(xmlbox.find('xmax').text)),
                     int(float(xmlbox.find('ymax').text)))
                sf.write(wd + "/data/Imgs/" + name + ' ' + ",".join([str(a) for a in b]) + ',' + str(cls_id))
                sf.write('\n')
                w_id += 1
        sf.close()

    print("apllied %s" % w_id + " annotations")


if __name__ == "__main__":

    convert_annotation()
