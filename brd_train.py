"""
Created on Sun Feb 11 09:34:44 2018
@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import brd_data_proc as dp
import tensorflow as tf
import designCNN as dc
import numpy as np
import random

"""define data dimensions"""
# raw image size
im_height = 1500
im_width = 1500
# image patch size
im_patch_height= 64
im_patch_width = 64
nChannel = 3
# label patch size
lb_patch_height= 16
lb_patch_width = 16
# number of raw images used for training
num_rawIm_trn = 0  # 0 means using all training images
num_rawIm_val = 0
# batch size
batch_size_trn = 32  # number of patches in this batch for training
batch_size_val = 1000  # number of patches in this batch for validation
# number of epochs for training
num_epochs = 3


"""Data Preprocessing: Normalization"""
# load raw images and labels for training and validation
im_trn, lb_trn = dp.load_data('./mass_buildings_roads_sub/train', num_rawIm_trn)
im_val, lb_val = dp.load_data('./mass_buildings_roads_sub/valid', num_rawIm_val)
print('Number of train images loaded: {0}'.format(len(im_trn)))
print('Number of valid images loaded: {0}'.format(len(im_val)))


# change labels data type to int32 and set 255 to be 1 s
# training data
lb_trn = np.array(lb_trn)
# validation data
lb_val = np.array(lb_val)

print('Label data type changed...')
#print(lb_norm_trn[0])

# compute the coordinates of patch center points
patch_cpt_trn = dp.patch_center_point(im_height,
                                      im_width,
                                      im_patch_height,
                                      im_patch_width,
                                      lb_patch_height,
                                      lb_patch_width,
                                      len(im_trn))
patch_cpt_val = dp.patch_center_point(im_height,
                                      im_width,
                                      im_patch_height,
                                      im_patch_width,
                                      lb_patch_height,
                                      lb_patch_width,
                                      len(im_val))

# delete null patches, i.e., patches with large number of pixels DN = 255
print('Patch Cleaning...')
patch_cpt_trn = dp.patch_clean(im_trn, patch_cpt_trn, im_patch_height, im_patch_width)
patch_cpt_val = dp.patch_clean(im_val, patch_cpt_val, im_patch_height, im_patch_width)

print('Patch center points generated...')


"""Design CNN for training"""
# create a graph to hold the CNN model
graph = tf.Graph()
# create a CNN model in the graph
with graph.as_default():
    # input layer
    x = tf.placeholder(tf.float32, [None, im_patch_height, im_patch_width, nChannel], name='x')
    # true label
    y_true = tf.placeholder(tf.int32, [None, lb_patch_height, lb_patch_width], name='y_true')

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
    num_filters_conv5 = 768

    size_avgpool = 3

    num_output_channels = 3

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
                              [-1, lb_patch_height, lb_patch_width, num_output_channels],
                              name='outpu_patch')

    # define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=output_patch)
    loss = tf.reduce_sum(cross_entropy, name='loss')  # summation of pixel-wise cross entropy
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    # get the prediction
    y_pred = tf.argmax(output_patch, axis=3, output_type=tf.int32, name='y_pred')

    # compute correct prediction
    correct = tf.equal(y_pred, y_true, name='correct')
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')

    # save the model
    saver = tf.train.Saver()

    # initialization
    init = tf.global_variables_initializer()

    print('CNN created.\n')


with tf.Session(graph = graph) as sess:
    print('\nStart training CNN.\n')
    # initialize CNN
    sess.run(init)

    sz_imtrain_trn = len(patch_cpt_trn)
    num_iterations = int(sz_imtrain_trn / batch_size_trn)

    num_patch_val = len(patch_cpt_val)
    num_iterations_val = int(num_patch_val / batch_size_val)

    acc_trn = []  # record training accuracy for each iteration/batch
    acc_trn_avg = [] # record average training accuracy for every N iteration
    acc_val_avg = []  # record average valid accuracy for every N iteration

    # extract all patches of valid dataset
#    im_patch_val, lb_patch_val = dp.data_patch(im_val,
#                                               lb_val,
#                                               patch_cpt_val,
#                                               im_patch_width,
#                                               im_patch_height,
#                                               lb_patch_width,
#                                               lb_patch_height)
#    # patch normalization
#    im_patch_val = [dp.image_normalize(im) for im in im_patch_val]

    for ep in range(num_epochs):
        # shuffle the training patches
        # patch_cpt is in fact the indicator of each patch
        random.shuffle(patch_cpt_trn)

        for it in range(num_iterations):
            # extract image and label patches in current batch for training data
            im_patch_batch_trn, lb_patch_batch_trn = dp.data_batch(im_trn,
                                                                   lb_trn,
                                                                   patch_cpt_trn,
                                                                   im_patch_width,
                                                                   im_patch_height,
                                                                   lb_patch_width,
                                                                   lb_patch_height,
                                                                   batch_size_trn, it)
            # patch normalization
            im_patch_batch_trn = [dp.image_normalize(im) for im in im_patch_batch_trn]

            # feed data
            Feed_Dict_Trn = {x: im_patch_batch_trn, y_true: lb_patch_batch_trn}  # only a batch of data patches

            # train CNN
            sess.run(train_op, feed_dict=Feed_Dict_Trn)
            # training accuracy
            _acc_t_ = sess.run(acc, feed_dict=Feed_Dict_Trn)
            acc_trn.append(_acc_t_)

            if it % 300 == 0:  # each image have about 8100/32 = 253 batches, after each image, estimate the accuracy
                # average training accuracy up to now
                _acc_t_avg_ = np.mean(np.array(acc_trn))
                acc_trn_avg.append(_acc_t_avg_)

                # validation accuracy
                # due to limited computer memory, we cannot feed all validation patches into the model at one time
                _acc_v_batch_ = []
                for it_v in range(num_iterations_val):
                    # extract image and label patches in current batch for training data
                    im_patch_batch_val, lb_patch_batch_val = dp.data_batch(im_val,
                                                                           lb_val,
                                                                           patch_cpt_val,
                                                                           im_patch_width,
                                                                           im_patch_height,
                                                                           lb_patch_width,
                                                                           lb_patch_height,
                                                                           batch_size_val, it_v)

                    # patch normalization
                    im_patch_batch_val = [dp.image_normalize(im) for im in im_patch_batch_val]

                    # feed data batch
                    Feed_Dict_Val = {x: im_patch_batch_val, y_true: lb_patch_batch_val}
                    # compute the accuracy of this valid batch
                    _acc_v_batch_.append(sess.run(acc, feed_dict=Feed_Dict_Val))

                # record the average accuracy of this <it>
                _acc_v_avg_ = np.mean(np.array(_acc_v_batch_))
                acc_val_avg.append(_acc_v_avg_)

                # save the trained model
                saver.save(sess, "./model_save/rd_model", global_step = ep*num_iterations+it)
                # every 300 iteration, validate
                msg = "Epoch: {0}...Iteration: {1}...Train acc: {2:>6.1%}...Valid acc: {3:>6.1%}"
                print(msg.format(ep, it, _acc_t_avg_, _acc_v_avg_))

    # write training and valid accuracy into file
    np.savetxt('acc_train.txt', np.array(acc_trn))
    np.savetxt('acc_train.txt', np.array(acc_trn_avg))
    np.savetxt('acc_valid.txt', np.array(acc_val_avg))

    print("\nTraining Done...")