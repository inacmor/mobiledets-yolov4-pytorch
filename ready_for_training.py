from utils.voc2yolo4 import voc_to_yolo
import xml.etree.ElementTree as ET
from os import getcwd

#======================参数===================================#
xmlfilepath=r'./data/Annotations/'
saveBasepath=r"./data/ImageSets/Main/"
classfilepath=r'./data/classes.txt'

sets=['train', 'val', 'test']

#======================参数===================================#

voc_to_yolo(xmlfilepath, saveBasepath)


# #读取类别
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


classes = get_classes(classfilepath)


def convert_annotation(image_id, list_file):
    in_file = open('./data/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

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
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    image_ids = open('./data/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('data_%s.txt'%(image_set), 'w')

    for image_id in image_ids:
        list_file.write('%s/data/Imgs/%s.jpg' % (wd, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
