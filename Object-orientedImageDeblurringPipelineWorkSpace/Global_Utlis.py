'''
Created on May 2, 2018

@author: erictseng
'''
from multiprocessing import Process
from matplotlib import pyplot as plt
import time
import os
from math import log10
import numpy as np
import pandas as pd
import torch
import math
import torchvision.utils as utils
from torch.autograd import Variable
from srgan_master import pytorch_ssim
from PIL import Image
from torchvision.transforms import Compose, ToTensor, RandomCrop, ToPILImage, Resize


        
def Print_CheckBound(string):
    print("==========================================================================================")
    print("** {} **".format(string))
    print("==========================================================================================")

def Print_EndCheckBound():
    print("==========================================================================================\n\n")

def crop(dic, i, rgb_image):
    image = rgb_image
    x0 = dic[i]["x"]
    y0 = dic[i]["y"]
    width = dic[i]["width"]
    height = dic[i]["height"]
    return image[y0:y0+height , x0:x0+width, :] 

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# bicubic upsampling
def bicubic_upsampling(image_size , upscale_factor): 
    return Compose([
        Resize(image_size * upscale_factor, interpolation=Image.BICUBIC)
    ])
    

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).data.mean()
    score = 10 * math.log10(1 / mse)
    return score
def ssim(img1, img2):
    score = pytorch_ssim.ssim(img1, img2).data[0]
    return score

def evaluation(test_image):
    
    tf_tensor=Compose([ToTensor()])
    
    hr_tensor = tf_tensor(hr_image).unsqueeze(0)
    lr_tensor = tf_tensor(test_image).unsqueeze(0)
    hr_variable = Variable(hr_tensor, volatile=True)
    lr_variable = Variable(lr_tensor, volatile=True)
    
    psnr_score = psnr(Variable(hr_tensor, volatile=True), Variable(lr_tensor, volatile=True))
    ssim_score = ssim(hr_variable,lr_variable)
    return psnr_score, ssim_score

def train_lr_transform(crop_size, upscale_factor): 
    return Compose([
        Resize((crop_size // upscale_factor, crop_size // upscale_factor), interpolation=Image.BICUBIC)
    ])
    
def train_hr_transform(crop_size, upscale_factor): 
    return Compose([
        Resize((crop_size , crop_size ), interpolation=Image.BICUBIC)
    ])
    
    

import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from srgan_master.model import Generator


def Apply_SRGAN(segmented_obj_names_ls, HOME_PATH_str):
    for segmented_image_name in segmented_obj_names_ls:
        print("This pipeline will apply SRGAN on single image.\n")
        parser = argparse.ArgumentParser(description='Test Single Image')
        parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
        parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
        parser.add_argument('--image_name', default = segmented_image_name, type=str, help='test low resolution image name')
        parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
        parser.add_argument('--dataset', default='VOC2012', type=str, help='dataset name')
        opt = parser.parse_args()
        print('**Default arguments are: {0}**\n\n'.format(opt))
        
        UPSCALE_FACTOR = opt.upscale_factor
        TEST_MODE = True if opt.test_mode == 'GPU' else False
        IMAGE_NAME = opt.image_name
        MODEL_NAME = opt.model_name
        dataset = opt.dataset
        SRGAN_PATH = HOME_PATH_str + 'Object-orientedDeblurringPipeline/srgan_master/'
        Pipeline_PATH = HOME_PATH_str + 'Object-orientedDeblurringPipeline/'
        
        model = Generator(UPSCALE_FACTOR).eval()
        if TEST_MODE:
            model.cuda()
            model.load_state_dict(torch.load(SRGAN_PATH + 'epochs/' + MODEL_NAME))
        else:
            model.load_state_dict(torch.load(SRGAN_PATH + 'epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
        
        #Apply SRGAN
        image = Image.open(Pipeline_PATH + 'Images/'+ IMAGE_NAME)
        image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
        if TEST_MODE:
            image = image.cuda()
            
        start = time.clock()
        out = model(image)
        elapsed = (time.clock() - start)
        print('cost' + str(elapsed) + 's')
        out_img = ToPILImage()(out[0].data.cpu())
        out_img.save(Pipeline_PATH + 'Images/' + '_out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
        print(IMAGE_NAME + 'DONE. \n **')
        
        
def calculate_bicubic_upsampling_size_h_w(ori_img,upscale_factor):
    return ori_img[0]*upscale_factor, ori_img[1]*upscale_factor

def BICUBIC_upsampling(h,w, upscale_factor):
    return Compose([ Resize((h, w), interpolation = Image.BICUBIC) ])


from skimage.io import imsave, imread
from skimage import img_as_float
from scipy.misc import imresize

def Bicubic_eric(img_in, factor = 400):
    img_double = img_as_float(img_in)
    if type(img_in) == np.ndarray:
        try: 
            img_out = imresize(img_in, size=factor, interp= 'bicubic')
            return img_out
        except:
            raise ValueError("Problems exist in Bicubic Resampling.")
    else:
        raise ValueError("The type of img_in should be np.ndarray. Current is {0}".format(type(img_in)))
        
def coords_format(coords_list):
    res_coords_list = []
    for coord_tuple in coords_list:
        mask = (int(coord_tuple[0][0]),int(coord_tuple[0][1]), int(coord_tuple[1]), int(coord_tuple[2]))
        res_coords_list.append(mask)
    return res_coords_list

def upsampled_coords_ls_format(formatted_coords_ls):
    return [(tpl[0]*4, tpl[1]*4, tpl[2]*4, tpl[3]*4) for tpl in formatted_coords_ls]

def object_transpose(obj_array):
    #return obj_array.transpose(1,0,2)
    return obj_array

def SR_obj_ls_PIL2array(SR_obj_ls):
    return [object_transpose(np.asarray(img)) for img in SR_obj_ls]
    

def check_and_resize_objects(SR_obj_array_ls, upsampled_coords_ls, factor=4):
    corped_SR_obj_array_ls = []
    for cnt in range(0, len(SR_obj_array_ls)):
        if SR_obj_array_ls[cnt].shape == (upsampled_coords_ls[cnt][3], upsampled_coords_ls[cnt][2], 3):
            corped_SR_obj_array_ls.append(SR_obj_array_ls[cnt])
        else:
            corped_SR_obj_array = SR_obj_array_ls[cnt][0:upsampled_coords_ls[cnt][3], 0:upsampled_coords_ls[cnt][2], :] #Corp
            corped_SR_obj_array_ls.append(corped_SR_obj_array)
            
    return corped_SR_obj_array_ls  


def Filling_Objects_into_upsampled_rgb_image(upsampled_rgb_image, corped_SR_obj_arrays_ls, upsampled_formatted_coords_ls):
    mask_rgb_image = upsampled_rgb_image.copy()
    for cnt in range(0, len(corped_SR_obj_arrays_ls)):
        startX = upsampled_formatted_coords_ls[cnt][0] 
        startY = upsampled_formatted_coords_ls[cnt][1]
        endX = upsampled_formatted_coords_ls[cnt][0]+upsampled_formatted_coords_ls[cnt][2]
        endY = upsampled_formatted_coords_ls[cnt][1]+upsampled_formatted_coords_ls[cnt][3]
        #print(startX, startY, endX, endY)
        
        mask_obj = corped_SR_obj_arrays_ls[cnt]
        mask_rgb_image[startY:endY, startX:endX, :] = mask_obj
    return mask_rgb_image

    
