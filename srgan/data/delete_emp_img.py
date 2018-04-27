import os


from PIL import Image
from os import listdir
from os.path import join

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

dataset_name = 'OOIS2012'
dataset_dir_val = dataset_name+'/val'
dataset_dir_train = dataset_name+'/train'

image_filenames_train = [join(dataset_dir_train, x) for x in listdir(dataset_dir_train) if is_image_file(x)]
image_filenames_val = [join(dataset_dir_val, x) for x in listdir(dataset_dir_val) if is_image_file(x)]

image_filenames = image_filenames_train + image_filenames_val

for i in range(len(image_filenames)):
	try:
		image = Image.open(image_filenames[i])
	except OSError:
		os.remove(image_filenames[i])
		print('{} has been removed'.format(image_filenames[i]))
