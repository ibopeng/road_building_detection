# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import dataProc as dp
import tensorflow as tf
import designCNN as dc
import random
import matplotlib.pyplot as plt

"""define data dimensions"""
# raw image size
im_width = 1500
im_height = 1500
# image patch size
im_patch_width = 64
im_patch_height= 64
nChannel = 3
# label patch size
lb_patch_width = 16  
lb_patch_height= 16
# number of raw images used for training
num_rawIm = 100
# batch size
batch_size = 32
# number of epochs for training
num_epochs = 5
  
"""Data Preprocessing: Normalization"""
# load raw images and labels
im_raw_trn, la_raw_trn = dp.load_data('./data/train', num_rawIm)
# normalization of raw images
im_norm_trn = [dp.image_normaliztion(im_raw_trn[i]) for i in range(num_rawIm)]   
# change labels data type to int32 and set 255 to be 1 
lb_norm_trn = dp.np.array(la_raw_trn)
lb_norm_trn = lb_norm_trn.astype(dp.np.int32)
lb_norm_trn /= 255
# compute the coordinates of patch center points
patch_cpt = dp.patch_ceter_point(im_width, 
                                 im_height, 
                                 im_patch_width, 
                                 im_patch_height, 
                                 lb_patch_width, 
                                 lb_patch_height, 
                                 num_rawIm)


"""Design CNN for training"""
# create a graph to hold the CNN model
graph = tf.Graph()
# create a CNN model in the graph
with graph.as_default():
    # input layer
    x = tf.placeholder(tf.float32, [None, im_patch_width, im_patch_height, nChannel], name='x')
    # true label
    y_true = tf.placeholder(tf.int32, [None, lb_patch_width, lb_patch_height], name='y_true')
    
    # archetecture of each CNN layer
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
    
    layer_cnn1 = dc.create_cnn_layer(input=x,
                                  num_input_channels = nChannel,
                                  num_cnn_filters = num_filters_conv1,
                                  size_cnn_filter = size_conv1,
                                  strides = [1,4,4,1])
                                  
    #max-pooling
    maxpool_layer_cnn1 = tf.nn.max_pool(value=layer_cnn1,
                               ksize=[1, size_maxpool, size_maxpool, 1],
                               strides=[1,1,1,1],
                               padding='VALID')                             
    
    layer_cnn2 = dc.create_cnn_layer(input=maxpool_layer_cnn1,
                                  num_input_channels = num_filters_conv1,
                                  num_cnn_filters = num_filters_conv2,
                                  size_cnn_filter = size_conv2,
                                  strides = [1,1,1,1])
                                  
    layer_cnn3 = dc.create_cnn_layer(input=layer_cnn2,
                                  num_input_channels = num_filters_conv2,
                                  num_cnn_filters = num_filters_conv3,
                                  size_cnn_filter = size_conv3,
                                  strides = [1,1,1,1])    
    
    layer_cnn4 = dc.create_cnn_layer(input=layer_cnn3,
                                  num_input_channels = num_filters_conv3,
                                  num_cnn_filters = num_filters_conv4,
                                  size_cnn_filter = size_conv4,
                                  strides = [1,1,1,1])                             
    
    layer_cnn5 = dc.create_cnn_layer(input=layer_cnn4,
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
                              [-1, lb_patch_width, lb_patch_height, num_output_channels],
                              name = 'outpu_patch')
    
    # define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_true, logits = output_patch)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    
    # get the prediction
    y_pred = tf.argmax(output_patch, dimension=3, output_type = tf.int32, name='y_pred')
    
    # compute correct prediction
    correct = tf.equal(y_pred, y_true, name='correct')
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
    
    saver = tf.train.Saver()   
    
    # initialization
    init = tf.global_variables_initializer()
    
"""Train CNN"""
sess = tf.Session(graph = graph)
# initialize variables
_ = sess.run(init)

sz_imtrain_trn = len(patch_cpt)
num_iterations = sz_imtrain_trn / batch_size

acc_trn = [] 
tmp = 0 
  
for ep in range(num_epochs):
    # shuffle the training patches
    # patch_cpt is in fact the indicator of each patch
    random.shuffle(patch_cpt)
    
    for it in range(num_iterations):
        # extract image and label patches in current batch
        im_patch_batch_trn, lb_patch_batch_trn = dp.data_batch(im_norm_trn, 
                                                               lb_norm_trn, 
                                                               patch_cpt, 
                                                               im_patch_width, 
                                                               im_patch_height, 
                                                               lb_patch_width, 
                                                               lb_patch_height, 
                                                               batch_size,
                                                               it)
        # feed data
        Feed_Dict_Trn = {x: im_patch_batch_trn, y_true: lb_patch_batch_trn}       
        # train CNN
        sess.run(train_op, feed_dict = Feed_Dict_Trn)
        
        if it % 10 == 0:
            #at the end of this epoch, compute the accuracy           
            acc_trn.append(sess.run(acc, feed_dict = Feed_Dict_Trn))
            msg = "Epoch: {0}...Train acc: {1:>6.1%}"
            print(msg.format(ep+1, acc_trn[tmp]))
            tmp = tmp + 1

sess.close()        
        
    

    



   
# test data loading
#i

#im_patch_batch, lb_patch_batch = train_batch(images, labels, patch_cpt, 64, 64, 16, 16, 60, 7)


#im_norm = image_normaliztion(images[1])
#plt.imshow(im_norm)
#plt.imshow(images[1])
#plt.show()