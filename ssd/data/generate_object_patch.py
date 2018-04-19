import os
import sys
import matplotlib.pyplot as plt
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)

from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import VOC_CLASSES as labels
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd


net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
save_dir = "/Users/Sean/Desktop/OOIS"
top_k=10
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = VOCDetection(VOC_ROOT, [('2012', 'trainval')], None, VOCAnnotationTransform())


def crop(dic, i, rgb_image):
    image = rgb_image
    x0 = dic[i]["x"]
    y0 = dic[i]["y"]
    width = dic[i]["width"]
    height = dic[i]["height"]
    return image[y0:y0+height , x0:x0+width, :]
    

number_of_image = len(testset)
for img_id in range(10):

    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    detections = y.data
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    coords_list = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            coords_list.append(coords)
            j+=1

    #{'a':(1 if i<2 else 4) for key, value in dic.items() }
    mydic = []
    for i in range(len(coords_list)):

        # x
        if int(coords_list[i][0][0])< 0:
            x = 0
        elif int(coords_list[i][0][0])>500:
            x = 500
        else:
            x = int(coords_list[i][0][0])

        # y
        if int(coords_list[i][0][1])< 0:
            y = 0
        elif int(coords_list[i][0][1])>375:
            y = 375
        else:
            y = int(coords_list[i][0][1])

        # width
        if int(coords_list[i][1])< 0:
            width = 0
        elif int(coords_list[i][1])>500:
            width = 500
        else:
            width = int(coords_list[i][1])

        # height
        if int(coords_list[i][2])< 0:
            height = 0
        elif int(coords_list[i][2])>375:
            height = 375
        else:
            height = int(coords_list[i][2])

        dic = {
        "height":height ,
        "width": width,
        "x": x,
        "y": y
        }
        mydic.append(dic)

    print ('img{}'.format(img_id), mydic)
    # plot
    #save_dir = "/Users/Sean/Desktop/OOIS"
    from PIL import Image
    import numpy as np
    for i in range(len(coords_list)):
        object_patch = crop(mydic, i, rgb_image)
        object_img = Image.fromarray(object_patch, 'RGB')
        object_img.save(save_dir + "/img_{}_obj_{}.jpg".format(img_id,i))
