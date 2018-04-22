import os
from os.path import isfile, join
from os import listdir
import numpy
import os.path as osp
import shutil, sys
import random

dir_list = './VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
dir_jpg = './VOCdevkit/VOC2012/JPEGImages/'
image_name_list = []
for line in open(dir_list):
    image_name_list.append(line.strip())   
number_of_train = int(len(image_name_list) * 0.7)
random.shuffle(image_name_list)
image_name_train = image_name_list[:number_of_train]
image_name_val = image_name_list[number_of_train:]


def image_full_path(image_name_list):
    image_path_list = []
    for image_id in range(len(image_name_list)):
        image_path = osp.join(dir_jpg, '%s.jpg' % image_name_list[image_id])
        image_path_list.append(image_path)
    return image_path_list
image_train_list = image_full_path(image_name_train)
image_val_list = image_full_path(image_name_val)


train_folder_dir = './VOC2012/train'
val_folder_dir = './VOC2012/val'
if not os.path.exists(train_folder_dir):
    os.makedirs(train_folder_dir)   
if not os.path.exists(val_folder_dir):
    os.makedirs(val_folder_dir)


for image in image_train_list:
    shutil.move(image, train_folder_dir)
for image in image_val_list:
    shutil.move(image, val_folder_dir)  