import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_9.pth', type=str, help='generator model epoch name')
parser.add_argument('--dataset', default='VOC2012', type=str, help='dataset name')
parser.add_argument('--modeldir', default='epochs/', type=str, help='model saved dir []')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
dataset = opt.dataset
modeldir = opt.modeldir

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(modeldir + dataset + '/' + dataset + '_'+ MODEL_NAME))
else:
    model.load_state_dict(torch.load(modeldir + dataset + '/' + dataset + '_'+ MODEL_NAME, map_location=lambda storage, loc: storage))


image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.clock()
out = model(image)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
if modeldir == 'wgan_epochs/':
    out_img.save('test_result/' + dataset + '_wgan_out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
else:
    out_img.save('test_result/' + dataset + '_out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
