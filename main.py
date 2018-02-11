# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:34:40 2017

@author: bo
"""
#neural network operation
import tensorflow as tf
#matrix operation
import numpy as np
import random
#directory operation
import os
#image operation
from skimage.io import imread
import matplotlib.pyplot as plt

from sklearn import preprocessing
from skimage import transform

#load image and label data
def load_data(data_directory):
    images = []
    labels = []
    
    #path of images and labels
    path_sat_image = os.path.join(data_directory, 'sat')
    path_map_label = os.path.join(data_directory, 'map')

    #get filenames for each image for label file
    image_names = []
    label_names = []
    filelist = os.listdir(path_sat_image)
    for f in filelist:
        if f.endswith('.tiff'):
            #image file names (include path)
            image_names.append(os.path.join(path_sat_image, f))
            #label file names = image file names but with different path
            #note that label data end with .tif rather than .tiff
#            fb = f[0:-1]
            label_names.append(os.path.join(path_map_label, f))
                              
    #read images and labels                          
    images = [imread(im) for im in image_names]
    labels = [imread(lb) for lb in label_names]
    
    return images, labels, filelist

#initilize weight
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
#initilize bias
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
    

#create CNN
def create_cnn_layer(input,
                     num_input_channels,
                     num_cnn_filters,
                     size_cnn_filter,
                     strides):
    
    #create weights
    weights = create_weights(shape=[size_cnn_filter, size_cnn_filter, num_input_channels, num_cnn_filters])
    
    #create bias
    biases = create_biases(size=num_cnn_filters)
    
    #create cnn layer
    cnn_layer = tf.nn.conv2d(input, 
                             filter=weights, 
                             strides=strides, 
                             padding='VALID')
    
    #add bias to the layer
    cnn_layer += biases
    
    return cnn_layer    

#get path for current running script    
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
#path for training data
data_path = os.path.join(ROOT_PATH, 'subtrain_3')

def image_normalization(images):
    images = np.array(images)
    images = images.astype(np.float32)
    #preprocessing: normalization
    images_norm = []
    for im in images:
        for band in range(0,3):
            im[:,:,band] = preprocessing.scale(im[:,:,band])
    #        im[:,:,band] = im_band_scale
        images_norm.append(im) 
    return images_norm

#load training data and labels
images, labels, image_filelist = load_data(data_path)
num_data = len(images)
image_size = 64
num_channels = 3
num_classes = 2

#normalie image patches
images_norm = image_normalization(images)

labels = np.array(labels)
labels = labels.astype(np.int32)
labels /= 255
label_size = 16
#labesl = tf.cast(labels, tf.int32)





#labels = tf.reshape(labels, shape=[None, label_size*label_size])

#Seperate data into "training set" and "validation set"
#total num_data = train + val
validation_size = 0.2
num_data_val = int(num_data * validation_size);
num_data_train = num_data - num_data_val;

train_data = images_norm[0:num_data_train]
train_label = labels[0:num_data_train]
val_data = images_norm[num_data_train:num_data]
val_label = labels[num_data_train:num_data]

sess = tf.Session()
#input image data
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels], name='x')
#labels
y_true = tf.placeholder(tf.int32, shape=[None, label_size, label_size], name='y_true')

#design CNN
size_conv1 = 12
num_filters_conv1 = 128

size_maxpool = 2

size_conv2 = 5
num_filters_conv2 = 256

size_conv3 = 3
num_filters_conv3 = 512

size_conv4 = 3
num_filters_conv4 = 32

size_conv5 = 3
num_filters_conv5 = 512

size_avgpool = 3

num_output_channels = 2
size_output_patch = label_size


layer_cnn1 = create_cnn_layer(input=x,
                              num_input_channels = num_channels,
                              num_cnn_filters = num_filters_conv1,
                              size_cnn_filter = size_conv1,
                              strides = [1,4,4,1])
                              
#max-pooling
maxpool_layer_cnn1 = tf.nn.max_pool(value=layer_cnn1,
                           ksize=[1, size_maxpool, size_maxpool, 1],
                           strides=[1,1,1,1],
                           padding='VALID')                             

layer_cnn2 = create_cnn_layer(input=maxpool_layer_cnn1,
                              num_input_channels = num_filters_conv1,
                              num_cnn_filters = num_filters_conv2,
                              size_cnn_filter = size_conv2,
                              strides = [1,1,1,1])
                              
layer_cnn3 = create_cnn_layer(input=layer_cnn2,
                              num_input_channels = num_filters_conv2,
                              num_cnn_filters = num_filters_conv3,
                              size_cnn_filter = size_conv3,
                              strides = [1,1,1,1])    

layer_cnn4 = create_cnn_layer(input=layer_cnn3,
                              num_input_channels = num_filters_conv3,
                              num_cnn_filters = num_filters_conv4,
                              size_cnn_filter = size_conv4,
                              strides = [1,1,1,1])                             

layer_cnn5 = create_cnn_layer(input=layer_cnn4,
                              num_input_channels = num_filters_conv4,
                              num_cnn_filters = num_filters_conv5,
                              size_cnn_filter = size_conv5,
                              strides = [1,1,1,1])

#global average pooling
avgpool_layer_cnn5 = tf.nn.avg_pool(value=layer_cnn5,
                                    ksize=[1, size_avgpool, size_avgpool, 1],
                                    strides=[1,1,1,1],
                                    padding='VALID')                             

#reshape the last layer: avgpool_layer_cnn5
output_patch = tf.reshape(avgpool_layer_cnn5, 
                          [-1, size_output_patch, size_output_patch, num_output_channels],
                          name = 'outpu_patch')


#predictions, calculate the prob for each pixel on three channels
#y_pred = tf.reshape(output_patch, shape=[None, size_output_patch*size_output_patch, num_output_channels])                    
#y_pred_cls = tf.argmax(y_pred, dimension=3)

sess.run(tf.global_variables_initializer())

#cost
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_true,
                                                               logits = output_patch)

cost = tf.reduce_mean(cross_entropy)

#optimization
train_opt = tf.train.AdamOptimizer(0.001).minimize(cost)

# Convert logits to label indexes
y_pred_cls = tf.argmax(output_patch, dimension=-1, output_type = tf.int32)

correct_pred = tf.equal(y_pred_cls, y_true)

acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.global_variables_initializer())

#feed batch data
#randomly select a batch of samples from a data set
def data_epoch(dataSet, labelSet, batchSize):
    dataEpochIdx = random.sample(range(len(dataSet)), batchSize)
    dataEpoch = [dataSet[i] for i in dataEpochIdx]
    labelEpoch = [labelSet[i] for i in dataEpochIdx]
    
    return dataEpoch, labelEpoch

def show_progress(epoch, FeedDictTrn, FeedDictVal, val_loss):
    Trn_accuracy = sess.run(acc, feed_dict = FeedDictTrn)
    Val_accuracy = sess.run(acc, feed_dict = FeedDictVal)
    
    msg = "Epoch: {0}...Train acc: {1:>6.1%}...Val acc: {2:>6.1%}...Val loss: {3:.3f}"
    print(msg.format(epoch, Trn_accuracy, Val_accuracy, val_loss))

epoch = 0
def train(num_epochs, batchSize):
    global epoch
    
    for i in range(num_epochs):
        dataEpochTrn, labelEpochTrn = data_epoch(train_data, train_label, batchSize)
        dataEpochVal, labelEpochVal = data_epoch(val_data, val_label, batchSize)
        
        FeedDictTrn = {x: dataEpochTrn, y_true: labelEpochTrn}
        FeedDictVal = {x: dataEpochVal, y_true: labelEpochVal}
        
        sess.run(train_opt, feed_dict=FeedDictTrn)
        
        val_loss = sess.run(cost, feed_dict=FeedDictVal)
        
        epoch += 1
        
        #show progress
        show_progress(epoch, FeedDictTrn, FeedDictVal, val_loss)
   
batchSize = 64     
train(50, batchSize)

#test image 
#path for testing images
test_path = os.path.join(ROOT_PATH, 'subtest_1')
#load testing data and labels
test_images, test_labels, test_filelist = load_data(test_path)
#normalize test images
test_images_norm = image_normalization(test_images)

test_labels = np.array(test_labels)
test_labels = test_labels.astype(np.int32)
test_labels /= 255

predicted = sess.run(y_pred_cls, feed_dict={x:test_images_norm})     

#small image patches merge to an big one
def image_joint(patch_list, patch_images):    
    #image size
    num_pathes = len(patch_list)
    num_row_patches = int(num_pathes ** 0.5)
    size_row = num_row_patches * 16
    size_col = size_row
    image = np.zeros((size_row, size_col), dtype=int32)
    
    for k in range(num_pathes):
        patch_name = patch_list[k]
        patch_row = int(patch_name[-10:-8])
        patch_col = int(patch_name[-7:-5])
        #patch origin pixel location
        i = patch_row * 16
        j = patch_col * 16
        image[i:i+16, j:j+16] = patch_images[k]
        
    return image
    
image = image_joint(test_filelist, predicted)
plt.imshow(image)