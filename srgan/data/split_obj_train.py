import os
from os.path import isfile, join
from os import listdir
import numpy
import os.path as osp
import shutil, sys
import random

# read all the image directory 
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
dataset_dir = './OOIS'
image_dir_list = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

# split train and val images
number_of_train = int(len(image_dir_list) * 0.7)
random.shuffle(image_name_list)
image_train_list = image_name_list[:number_of_train]
image_val_list = image_name_list[number_of_train:]

# create new train and val folders
train_folder_dir = './OOIS2012/train'
val_folder_dir = './OOIS2012/val'
if not os.path.exists(train_folder_dir):
    os.makedirs(train_folder_dir)   
if not os.path.exists(val_folder_dir):
    os.makedirs(val_folder_dir)
 
#move the image to train and val folders 
for image in image_train_list:
    shutil.move(image, train_folder_dir)
for image in image_val_list:
    shutil.move(image, val_folder_dir)