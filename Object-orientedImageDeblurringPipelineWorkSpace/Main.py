"""                                        
   ____      ____      _____.    _____.     _____   __.  _______
  /  _  \   /  _  \   / ____ \  / ____ \   /  __ \ |. | |. ___. \
 |  / \  | |  / \  | | \_____  | \_____   |  |__| ||. |.| |___| |
 | |   | |.|.|. |. |  \_____ \  \_____ \. |. _. _/ |. |.|  _____/
 |  \_/  | |  \_/  |  ______/|. ______/ | |. |\.\  |. |.|.|
  \_____/   \_____/   \______/  \_______/ |__| \_\.|__|.|_|
"""                                            
#Author: Xinsheng(Sean) Zhang(xz1757@nyu.edu) Binqian(Eric) Zeng(bz866@nyu.edu)


#Import 
import argparse
import os
import sys
from click._compat import raw_input
from matplotlib.pyplot import subplot
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from multiprocessing import Process
from ssd_master.data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
from Global_Utlis import *
import warnings
warnings.filterwarnings("ignore")
#from VDSR_sharpen import *
from srgan_master import *
from ssd_master.ssd import build_ssd
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--HOME_PATH", default = "/Users/erictseng/Object-orientedImageDeblurringPipelineWorkSpace", type=str, help="Pipeline workspace PATH")
parser.add_argument("--SSD_weights_PATH", default = "./ssd_master/weights/ssd300_mAP_77.43_v2.pth", type=str, help="pre-trained SSD model weights PATH")
parser.add_argument("--size_input_images", default = 300, type=int, help="Size of Input Images") #Ref: Demo in https://github.com/amdegroot/ssd.pytorch
parser.add_argument("--num_classes_of_objects", default = 21, type=int, help="The number of object classes to score") #Ref: Demo in https://github.com/amdegro
parser.add_argument("--Image_ID", type=int, required=True, help="The Index of target image in the dataset(PascalVOC). In the demo, the index appears at the top of Image Document Name")
parser.add_argument("--Top_n_objects2Segment", default=61, type=int, help="Maximum of Number of Objects to be Segemented. ")
parser.add_argument("--cuda", default=0, type=int, help="0 means on CPU, 1 means on GPU.")
parser.add_argument("--Bicubic_UpSampling_Factor", default=400, type=int, help="Upsampling Factor for Bicubic Interpolation; 400 means zoom in x4")
args = parser.parse_args()

class ansi:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

print("""{}   {}**Objected-Oriented Super-Resolution Imaging Pipeline(OOSSRIP)**{}
  - Author: Xinsheng(Sean) Zhang(xz1757@nyu.edu) Binqian(Eric) Zeng(bz866@nyu.edu){}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

Print_EndCheckBound()
input("Press Enter to continue ..... ")
Print_EndCheckBound()

#if args.Image_ID not in [61, 73, 80,82, 108, 115, 139, 140, 141, 168, 174, 175, 189, 194]:
if args.Image_ID not in [61, 73, 115, 108]:
    raise ValueError("Selected Image_ID for Demo must be one of [61, 73, 115, 108]")


net = build_ssd('test', args.size_input_images, args.num_classes_of_objects) #Initialize SSD
net.load_weights(args.SSD_weights_PATH)

testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
img_id = args.Image_ID
image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

Print_CheckBound("The Original Image is Shown as Below.")
plt.figure(figsize=(10,10))
plt.imshow(rgb_image)
#plt.plot(rgb_image)
plt.pause(1.000)
raw_input("Press Enter to continue....")
Print_EndCheckBound()
plt.close()

#Take Original Image Input
orig_rgb_image = Image.fromarray(rgb_image, 'RGB')
orig_rgb_image.save('./Demo_Results/{0}_ori_image.jpg'.format(img_id))

#Pre-processing the Input
x = cv2.resize(image, (args.size_input_images, args.size_input_images)).astype(np.float32)
x -= (104.0, 117.0, 123.0) #Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
#plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)

#SSD Forward Pass
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if args.cuda == 1:
    xx = xx.cuda()
y = net(xx)

#Parse the Detections and View Results¶
from ssd_master.data import VOC_CLASSES as labels
top_k=args.Top_n_objects2Segment

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1

detections = y.data
detections.size()
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

detections = y.data
coords_list = []
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        #print (pt)
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        coords_list.append(coords)
        #print(coords)
        color = colors[i]
        #print (color)
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1

Print_CheckBound("Coordinates of Segementation Bounding Boxes are listed below: \n {0} \n\n".format(coords_list))
mydic = []
for i in range(len(coords_list)):   #Format coordinates of objects 
    dic = {
    "height": int(coords_list[i][2]),
    "width": int(coords_list[i][1]),
    "x": int(coords_list[i][0][0]),
    "y": int(coords_list[i][0][1])
    }
    mydic.append(dic)

#Show all segemented objects
#Plot_Segemented_Objects(coords_list, rgb_image, mydic)
subplot_cnt = 1
obj_cnt = 0
plt.imshow(rgb_image)
#rgb_image_2_4save = Image.fromarray(rgb_image, 'RGB')
#rgb_image_2_4save.save('./Images/CroppedImage.jpg')
#input("Coordinates of segement boxes are listed above; Then press Enter to continue ...\n\n")
subplot_cnt += 1
segmented_obj_ls = []
segmented_obj_names_ls  = []

while obj_cnt < len(coords_list):
    obj_patch = crop(mydic, obj_cnt, rgb_image)
    segmented_obj_ls.append(obj_patch)
    obj_cnt += 1
    obj_patch4save = Image.fromarray(obj_patch, 'RGB')#Save all segemented objects
    obj_patch4save.save('./Demo_Results/{0}_Object{1}.jpg'.format(img_id, obj_cnt))
    segmented_obj_names_ls.append('{0}_Object{1}.jpg'.format(img_id, obj_cnt))



mydic = []
for i in range(len(coords_list)):   #Format coordinates of objects 
    dic = {
    "height": int(coords_list[i][2]),
    "width": int(coords_list[i][1]),
    "x": int(coords_list[i][0][0]),
    "y": int(coords_list[i][0][1])
    }
    mydic.append(dic)

#Show all segemented objects
#Plot_Segemented_Objects(coords_list, rgb_image, mydic)
subplot_cnt = 1
obj_cnt = 0
#rgb_image_2_4save.save('./Images/CroppedImage.jpg')
#input("Coordinates of segement boxes are listed above; Then press Enter to continue ...\n\n")
subplot_cnt += 1
segmented_obj_ls = []
segmented_obj_names_ls  = []

while obj_cnt < len(coords_list):
    obj_patch = crop(mydic, obj_cnt, rgb_image)
    segmented_obj_ls.append(obj_patch)
    obj_cnt += 1
    obj_patch4save = Image.fromarray(obj_patch, 'RGB')#Save all segemented objects
    obj_patch4save.save('./Demo_Results/{0}_Object{1}.jpg'.format(img_id, obj_cnt))
    segmented_obj_names_ls.append('{0}_Object{1}.jpg'.format(img_id, obj_cnt))

segemented_rgb_image = rgb_image.copy()

#Apply SRGAN on all image
#Apply_SRGAN(segmented_obj_names_ls, HOME_PATH_str)
Print_CheckBound("\nSRGAN should be implemented on Segemented Objects in this step. \n\n  Demo loads HR Segemented Objects Results for time saving.\n")
input("Press Enter to contine......")
Print_EndCheckBound()

#Load Results for Demo
ori_rgb_image = Image.open("./Demo_Results/{0}_ori_image.jpg".format(img_id))
SR_obj_name_ls = ['others_out_srf_4_' + n for n in segmented_obj_names_ls]
SR_obj_ls = []
for i, result_obj_name in enumerate(SR_obj_name_ls):
    mask = Image.open('./Demo_Results/' + result_obj_name)
    SR_obj_ls.append(mask)

#Bicubic Upsampling
upsampled_rgb_image = Bicubic_eric(rgb_image, factor=args.Bicubic_UpSampling_Factor)
upsampled_rgb_image_save = Image.fromarray(upsampled_rgb_image)
upsampled_rgb_image_save.save("./Demo_Results/Upsampled_image_without_filling_{0}.jpg".format(img_id))

#Show Upsampled Image
Print_CheckBound("Check Bicubic upsampled Image. Upsampling Factor is {0}.".format(args.Bicubic_UpSampling_Factor/100))
#plt.imshow(upsampled_rgb_image)
plt.pause(1.000)
input("Press Enter to continue.......")
plt.close()
Print_EndCheckBound()

#Display Objects to be filled
Print_CheckBound("Objects to be filled are displayed below.")
for SR_obj in SR_obj_ls:
    plt.imshow(SR_obj)
    plt.pause(1.000)
input("Press Enter to continue......")
plt.close()
Print_EndCheckBound()

#Format before filling
formatted_coords_ls = coords_format(coords_list)
upsampled_formatted_coords_ls = upsampled_coords_ls_format(formatted_coords_ls)
#

#print(coords_list, formatted_coords_ls, upsampled_formatted_coords_ls)
SR_obj_arrays_ls = SR_obj_ls_PIL2array(SR_obj_ls)
#for obj in SR_obj_arrays_ls:
#    print(obj.shape)
corped_SR_obj_arrays_ls = check_and_resize_objects(SR_obj_arrays_ls, upsampled_formatted_coords_ls)
#for obj in corped_SR_obj_arrays_ls:
#    print(obj.shape)

#Filling
filled_rgb_image_array = Filling_Objects_into_upsampled_rgb_image(upsampled_rgb_image, corped_SR_obj_arrays_ls, upsampled_formatted_coords_ls)
#plt.imshow(Image.fromarray(filled_rgb_image_array))
#plt.pause(1.000)
#input("Here is the image with HR objects but sharpened")
#plt.close()
#Save Filled HR Image with out sharpening 
filled_rgb_image = Image.fromarray(filled_rgb_image_array)
filled_rgb_image.save("./Demo_Results/HR_Filled_withoutSharpen_{}.jpg".format(img_id))

Print_CheckBound("Objects have all been merged into the upsampled background. \n Show the image with HR objects.")
plt.imshow(filled_rgb_image)
plt.pause(1.000)
input("Press Enter to continue ....... ")
Print_EndCheckBound()
plt.close()

Print_CheckBound("Start Applying Neural Enhance")
#subprocess.call("apply_neural_enhance.sh", shell=True)
Print_EndCheckBound()
input("Press Enter to continue ....... ")
Print_EndCheckBound()


Print_CheckBound("Show the Final Result; Demo will load results to show")
Final_res = Image.open("./Demo_Results/HR_Filled_withoutSharpen_{}_ne1x.png".format(img_id))
plt.imshow(Final_res)
plt.pause(1.000)
input("Press Enter to contine ....... ")
Print_EndCheckBound()
plt.close()

print("Job on Image {} Done!!!!!!!\n\n".format(img_id))



	

