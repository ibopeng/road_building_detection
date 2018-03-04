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


def image_normalize(im):
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
    height, width, band = np.shape(im)
    #reshape for normalization    
#    im = np.reshape(im, (height*width, band))
    # preprocessing, normalization
    im_norm = []
    for i in range(band):
        im_bi = im[:, :, i]  # the ith band image
        im_bi = np.reshape(im_bi, height*width)  # change the single band image to be an 1d array
        mean_bi = np.mean(im_bi)  # sample mean
        std_bi = np.std(im_bi)  # standard deviation
        im_bi = (im_bi - mean_bi) / std_bi  # mean 0 and unit variance
#        im_bi = preprocessing.scale(im_bi)
#        im_bi = np.reshape(im_bi, [height, width])
        im_norm.append(im_bi)

    im_norm = np.array(im_norm)
    im_norm = im_norm.transpose()
    im_norm = np.reshape(im_norm, (height, width, band))

#    im_norm = preprocessing.scale(im, axis=0)

    """
    test
    """
#    xm = np.mean(im[0,:])
#    xs = np.std(im[0,:])
#    imt = im[0,:] - xm
#    imt /= xs


    # reshape to get normalized image
#    im_norm = np.reshape(im_norm, (height, width, band))
#    height, width, band = np.shape(im_norm)
    
    return im_norm

def patch_center_point(im_height, im_width, im_patch_height, im_patch_width, lb_patch_height, lb_patch_width, nIm):
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
    num_patch_height = int((im_height - im_patch_height) / lb_patch_height + 1)  # number of patches in row direction
    num_patch_width = int((im_width - im_patch_width) / lb_patch_width + 1)  # number of patches in column direction

    for k in range(nIm):
        for i in range(num_patch_height):
            cpt_i = int(im_patch_height / 2) + i * lb_patch_height # coordinate: row
            for j in range(num_patch_width):                
                cpt_j = int(im_patch_width / 2) + j * lb_patch_width # coordinate: column
                patch_cpt.append([k, cpt_i, cpt_j])
                
    return patch_cpt

#patch_cpt = patch_center_point(1500, 1500, 64, 64, 16, 16, 100)
#print(patch_cpt)

def data_batch(image, label, patch_cpt, im_patch_height, im_patch_width, lb_patch_height, lb_patch_width, batch_size, batch_idx):
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
    idx_patch_end = idx_patch_start + batch_size

    # the last batch may have size smaller than the batch_size
    if idx_patch_end > len(patch_cpt):
        idx_patch_end = len(patch_cpt)

    im_patch_batch = [] # array to store images patches in current batch
    lb_patch_batch = [] # array to store label patches in current batch

    for i in range(idx_patch_start, idx_patch_end):
        # add each image patch to [im_patch_batch]
        idx_im = patch_cpt[i][0] # index of raw image that the ith patch belongs to
        # coordinate system: [row, col]
        row_patch_cpt = patch_cpt[i][1] # row coordinate of center point
        col_patch_cpt = patch_cpt[i][2] # column ...
        
        # note that image patch and label patch have different sizes
        row_up_im = row_patch_cpt - int(im_patch_height / 2) # row index: upper most
        col_left_im = col_patch_cpt - int(im_patch_width / 2) # column index: left most
        row_up_lb = row_patch_cpt - int(lb_patch_height / 2)
        col_left_lb = col_patch_cpt - int(lb_patch_width / 2)
        
        # extract the patch
        im_patch = image[idx_im][row_up_im: row_up_im+im_patch_height, col_left_im: col_left_im+im_patch_width] # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][row_up_lb: row_up_lb+lb_patch_height, col_left_lb: col_left_lb+lb_patch_width]
        
        # add this patch to the batch array
        im_patch_batch.append(im_patch)
        lb_patch_batch.append(lb_patch)
    
    return im_patch_batch, lb_patch_batch

def data_patch(image,
               label,
               patch_cpt,
               im_patch_height,
               im_patch_width,
               lb_patch_height,
               lb_patch_width):
    """
    transform raw images into small patches
    """
    im_patch_all = []
    lb_patch_all = []

    for i in range(len(patch_cpt)):
        idx_im = patch_cpt[i][0] # index of the raw image this patch belongs to
        # coordinate system: [row, col]
        row_patch_cpt = patch_cpt[i][1] # row coordinate of center point
        col_patch_cpt = patch_cpt[i][2] # column ...

        # note that image patch and label patch have different sizes
        row_up_im = row_patch_cpt - int(im_patch_height / 2) # row index: upper most
        col_left_im = col_patch_cpt - int(im_patch_width / 2) # column index: left most
        row_up_lb = row_patch_cpt - int(lb_patch_height / 2)
        col_left_lb = col_patch_cpt - int(lb_patch_width / 2)


        # extract the patch
        im_patch = image[idx_im][row_up_im: row_up_im+im_patch_height, col_left_im: col_left_im+im_patch_width] # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][row_up_lb: row_up_lb+lb_patch_height, col_left_lb: col_left_lb+lb_patch_width]
        
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
        # coordinate system: [row, col]
        row_patch_cpt = patch_cpt[i][1]  # row coordinate of center point
        col_patch_cpt = patch_cpt[i][2]  # column ...

        # note that image patch and label patch have different sizes
        row_up_im = row_patch_cpt - int(im_patch_height / 2)  # row index: upper most
        col_left_im = col_patch_cpt - int(im_patch_width / 2)  # column index: left most
        row_up_lb = row_patch_cpt - int(lb_patch_height / 2)
        col_left_lb = col_patch_cpt - int(lb_patch_width / 2)

        # extract the patch
        im_patch = image[idx_im][row_up_im: row_up_im + im_patch_height, col_left_im: col_left_im + im_patch_width]  # in python, [a:b] does not include b but less than b
        lb_patch = label[idx_im][row_up_lb: row_up_lb + lb_patch_height, col_left_lb: col_left_lb + lb_patch_width]
        
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

    num_batch, batch_size, patch_height, patch_width = np.shape(output_patch)
    output_label = np.zeros((1500, 1500), dtype=np.int32)

    output_patch = np.reshape(output_patch, (num_batch*batch_size, patch_height, patch_width))

    # number of patches
    num_patch = num_batch*batch_size
    
    # assign patches to the big output label image
    for k in range(num_patch):
        cpt_row = patch_cpt[k][1]
        cpt_col = patch_cpt[k][2]
        output_label[cpt_row-8:cpt_row+8, cpt_col-8:cpt_col+8] = output_patch[k]

    return output_label
    