# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:34:44 2018

@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import os
import random
import numpy as np
from skimage.io import imread
from sklearn import preprocessing



def load_data(data_directory, nRand_sel):
    """
    load image and label data
    INPUT:
      data_directory: [string] path to load the original data
      nRand_sel: [integer], randomly select n samples from the loaded data in case the memory limitation
                 if all the data will be used, set nRand_sel = non-positive value (e.g., 0 or -1)
    OUTPUT:
      images: [Array] loaded images subset with size of nRand_sel
      labels: [Array] loaded truth map corresponding to [images]
    """

    images = []
    labels = []
    
    #path of images and labels
    path_sat_image = os.path.join(data_directory, 'sat')
    path_map_label = os.path.join(data_directory, 'map')

    #get filenames for each image for label file
    image_names = []
    label_names = []
    filelist = os.listdir(path_sat_image)
    filelist_sub = []

    # randomly select a subset of n sampels for further processing
    if nRand_sel < len(filelist) and nRand_sel > 0:
        filelist_subidx = random.sample(range(0, len(filelist)), nRand_sel)
        filelist_sub = [filelist[idx] for idx in filelist_subidx]
    else:
        filelist_sub = filelist
    
    for f in filelist_sub:
        if f.endswith('.tiff'):
            #image file names (include path)
            image_names.append(os.path.join(path_sat_image, f))
            #label file names = image file names but with different path
            
            fb = f[0:-1] #note that label data end with .tif rather than .tiff
            label_names.append(os.path.join(path_map_label, fb))
                              
    #read images and labels                          
    images = [imread(im) for im in image_names]
    labels = [imread(lb) for lb in label_names]
    
    return images, labels
    


def image_normaliztion(im):
    """
    Image normalization
    INPUT:
      im
    OUTPUT:
      im_norm
    """  
    # change the data type for nomalization    
    im = np.array(im)
    im = im.astype(np.float32)
    
    #size of im
    width, height, band = np.shape(im)
    #reshape for normalization    
    im = np.reshape(im, (width*height, band))      
    # preprocessing, normalization, mean 0 and unit variance
    im_norm = preprocessing.scale(im)    
    # reshape to get normalized image
    im_norm = np.reshape(im_norm, (width, height, band))
    
    return im_norm

def patch_center_point(im_width, im_height, im_patch_width, im_patch_height, lb_patch_width, lb_patch_height, nIm):
    """
    Compute the center point coordinates of the patch in raw images
    INPUT:
      im_width, im_height: width and height of raw images
      patch_width, patch_height: width and height of image patches
      nIm: number of raw images
    OUTPUT:
      A 3D array indicating the pathc center coordinates of each patch
    """
    # for each image, the pathc center coordinates are the same
    patch_cpt = []
    num_patch_width = int((im_width - im_patch_width) / lb_patch_width + 1)
    num_patch_height = int((im_height - im_patch_height) / lb_patch_height + 1)
    for k in range(nIm):
        for i in range(num_patch_height):
            cpt_i = int(im_patch_height / 2) + i * lb_patch_height # coordinate: y from up to bottom
            for j in range(num_patch_width):                
                cpt_j = int(im_patch_width / 2) + j * lb_patch_width # coordinate: x from left to right
                patch_cpt.append([k, cpt_j, cpt_i])
                
    return patch_cpt

#patch_cpt = patch_center_point(1500, 1500, 64, 64, 16, 16, 100)
#print(patch_cpt)

def data_batch(image, 
            label, 
            patch_cpt, 
            im_patch_width, 
            im_patch_height, 
            lb_patch_width, 
            lb_patch_height, 
            batch_size, 
            batch_idx):
    """
    Extract image patches for training according to patch_cpt
    Each images patches are overlapped to make label patches exactly join together without overlap or gap
    INPUT:
        image: raw images loaded
        label: road labels loaded
        patch_cpt: center points for each patch
        im/lb_patch_width/height: size of image/label patches
        batch_size: the number of patches used for training in one iteration
        batch_idx: the index of current batch
    OUTPUT:
        batch images with corresponding labels of size batch_size
    """
    idx_patch_start = batch_size * batch_idx # starting index of the first image patch in current patch
    idx_patch_end = idx_patch_start+batch_size
    # the last batch may have size greater than the batch_size
    if idx_patch_end > len(patch_cpt):
        idx_patch_end = len(patch_cpt)
    im_patch_batch = [] # array to store images patches in current batch
    lb_patch_batch = [] # array to store label patches in current batch
    

    for i in range(idx_patch_start, idx_patch_end):
        # add each image patch to [im_patch_batch]
        idx_im = patch_cpt[i][0] # index of raw images
        # coordinate system: x means pixel moving from left to right, y means pixel moving from up to bottom
        x_patch_cpt = patch_cpt[i][1] # x coordinate of center point
        y_patch_cpt = patch_cpt[i][2] # y ...
        
        # note that image patch and label patch have different sizes
        x_left_im = x_patch_cpt - int(im_patch_width / 2) # left most coordinate of the patch
        y_up_im   = y_patch_cpt - int(im_patch_height / 2) # upper most coordinate of the patch
        x_left_lb = x_patch_cpt - int(lb_patch_width / 2)
        y_up_lb   = y_patch_cpt - int(lb_patch_height / 2)
        
        # extract the patch
        im_patch = image[idx_im][x_left_im:x_left_im+im_patch_width, y_up_im:y_up_im+im_patch_height] # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][x_left_lb:x_left_lb+lb_patch_width, y_up_lb:y_up_lb+lb_patch_height]
        
        # add this patch to the array
        im_patch_batch.append(im_patch)
        lb_patch_batch.append(lb_patch)
    
    return im_patch_batch, lb_patch_batch

def data_patch(image,
            label, 
            patch_cpt, 
            im_patch_width, 
            im_patch_height, 
            lb_patch_width, 
            lb_patch_height):
    """
    transform raw images into small patches
    """
    im_patch_all = []
    lb_patch_all = []

    for i in range(len(patch_cpt)):
        idx_im = patch_cpt[i][0] # index of raw images
        # coordinate system: x means pixel moving from left to right, y means pixel moving from up to bottom
        x_patch_cpt = patch_cpt[i][1] # x coordinate of center point
        y_patch_cpt = patch_cpt[i][2] # y ...

        # note that image patch and label patch have different sizes
        x_left_im = x_patch_cpt - int(im_patch_width / 2) # left most coordinate of the patch
        y_up_im   = y_patch_cpt - int(im_patch_height / 2) # upper most coordinate of the patch
        x_left_lb = x_patch_cpt - int(lb_patch_width / 2)
        y_up_lb   = y_patch_cpt - int(lb_patch_height / 2)

        # extract the patch
        im_patch = image[idx_im][x_left_im:x_left_im+im_patch_width, y_up_im:y_up_im+im_patch_height] # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][x_left_lb:x_left_lb+lb_patch_width, y_up_lb:y_up_lb+lb_patch_height]
        
        # add this patch to the array
        im_patch_all.append(im_patch)
        lb_patch_all.append(lb_patch)
    
    return im_patch_all, lb_patch_all

def data_patch_batch_random(image,
                            label, 
                            patch_cpt, 
                            im_patch_width, 
                            im_patch_height, 
                            lb_patch_width, 
                            lb_patch_height,
                            batch_size):
    """
    transform raw images into small patches
    """
    im_patch_batch = []
    lb_patch_batch = []

    patch_idx = random.sample(range(len(patch_cpt)), batch_size)

    for i in patch_idx:
        idx_im = patch_cpt[i][0] # index of raw images
        # coordinate system: x means pixel moving from left to right, y means pixel moving from up to bottom
        x_patch_cpt = patch_cpt[i][1] # x coordinate of center point
        y_patch_cpt = patch_cpt[i][2] # y ...

        # note that image patch and label patch have different sizes
        x_left_im = x_patch_cpt - int(im_patch_width / 2) # left most coordinate of the patch
        y_up_im   = y_patch_cpt - int(im_patch_height / 2) # upper most coordinate of the patch
        x_left_lb = x_patch_cpt - int(lb_patch_width / 2)
        y_up_lb   = y_patch_cpt - int(lb_patch_height / 2)

        # extract the patch
        im_patch = image[idx_im][x_left_im:x_left_im+im_patch_width, y_up_im:y_up_im+im_patch_height] # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][x_left_lb:x_left_lb+lb_patch_width, y_up_lb:y_up_lb+lb_patch_height]
        
        # add this patch to the array
        im_patch_batch.append(im_patch)
        lb_patch_batch.append(lb_patch)
    
    return im_patch_batch, lb_patch_batch

def image_mosaic(output_patch, patch_cpt):
    """
    mosaic image patches into one image
    """

    output_patch = np.array(output_patch)
    output_patch = output_patch.astype(np.int32)

    num_batch, batch_size, patch_width, patch_height = np.shape(output_patch)
    output_label = np.zeros((1500, 1500), dtype=np.int32)

    output_patch = np.reshape(output_patch, (num_batch*batch_size, patch_width, patch_height))

    # number of patches
    num_patch = num_batch*batch_size
    
    # assign patches to the big output label image
    for k in range(num_patch):
        cpt_x = patch_cpt[k][1]
        cpt_y = patch_cpt[k][2]
        output_label[cpt_y-8:cpt_y+8, cpt_x-8:cpt_x+8] = output_patch[k]

    return output_label
    