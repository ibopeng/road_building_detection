#cut data into subsets of size 64_64

#neural network operation
import tensorflow as tf
#matrix operation
import numpy as np
#directory operation
import os
#image operation
from skimage import io
import matplotlib.pyplot as plt

#load raw image and label data
def load_data(data_directory):
    images = []
    labels = []
    
    #path of images and labels
    path_sat_image = os.path.join(data_directory, 'sat')
    path_map_label = os.path.join(data_directory, 'map')

    #get filenames for each image for label file
    image_names = []
    label_names = []
    for f in os.listdir(path_sat_image):
        if f.endswith('.tiff'):
            #image file names (include path)
            image_names.append(os.path.join(path_sat_image, f))
            #label file names = image file names but with different path
            #note that label data end with .tif rather than .tiff
            fb = f[0:-1]
            label_names.append(os.path.join(path_map_label, fb))
                              
    #read images and labels                          
    images = [io.imread(im) for im in image_names]
    labels = [io.imread(lb) for lb in label_names]
    
    return images, labels
    
#get path for current running script    
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
#path for training data
train_path = os.path.join(ROOT_PATH, 'test')
#path for validation data
#valid_path = os.path.join(ROOT_PATH, 'valid')

#load training data and labels
image_train, label_train = load_data(train_path)

raw_im_size = 1500
num_subset = (raw_im_size-64) / 16

#path for subsetted training data
path_subset_train = os.path.join(ROOT_PATH, 'subtest_1')
path_subset_train_img = os.path.join(path_subset_train, 'sat')
path_subset_train_label = os.path.join(path_subset_train, 'map')


#subset training images
nSample = 0
for k in range(len(image_train)):
    im = image_train[k]
    lb = label_train[k]
    nSample += 1
    for i in range(num_subset):
        #rows i
        imstart_i = 16*i
        lbstart_i = 24+16*i
        for j in range(num_subset):
            #columns j
            #image subset
            imstart_j = 16*j
            im_sub = im[imstart_i:imstart_i+64, imstart_j:imstart_j+64, :]
            #label subset            
            lbstart_j = 24+16*j
            lb_sub = lb[lbstart_i:lbstart_i+16, lbstart_j:lbstart_j+16]

            #creat the same name for image and label subset
            subset_name = "{}_{}_{}.tiff".format(nSample, i, j)
            #save image subset
            im_path = os.path.join(path_subset_train_img, subset_name)
            io.imsave(im_path, im_sub)
            #save label subset
            lb_path = os.path.join(path_subset_train_label, subset_name)
            io.imsave(lb_path, lb_sub)
            
    print("{}th image".format(nSample))

print("DONE...")