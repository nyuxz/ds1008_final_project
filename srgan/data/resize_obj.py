from PIL import Image
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, Resize
import torch
import os
from os import listdir
from os.path import join
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

dataset_name = 'OOIS2012'
dataset_dir = dataset_name+'/val'
image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

crop_size = 55
def small_obj_transform():
    return Compose([
        ToPILImage(),
        Resize(crop_size, interpolation=Image.BICUBIC)
    ])
small2big = small_obj_transform()

try:
    for i in range(len(image_filenames)):
        npim = cv2.imread(image_filenames[i])
        if npim.shape[1] < crop_size or npim.shape[0] < crop_size:
            image_tmp = small2big(npim)
            print ('{} has been resized'.format(image_filenames[i]))
            image_tmp.save(image_filenames[i])
except AttributeError:
    pass





