"""
This script is designed for testing the model trained by rd_train.py

Created on Sun Feb 16 2018
@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import brd_data_proc as dp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave

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
num_rawIm_test = 1
# batch size
batch_size_test = int((im_width - im_patch_width) / lb_patch_width + 1)

"""Data Preprocessing: Normalization"""
# load raw images and labels for testing
im_test, lb_test = dp.load_data('./mass_buildings_roads/test', num_rawIm_test)
print('Number of test images loaded: ', num_rawIm_test)

# change labels data type to int32 and set 255 to be 1 s
# training data
lb_test = np.array(lb_test)
lb_test = lb_test.astype(np.int32)
lb_test = [lb / 255 for lb in lb_test]
print('Label data type changed.')

# compute the coordinates of patch center points
patch_cpt_test = dp.patch_center_point(im_height, im_width, im_patch_height, im_patch_width, lb_patch_height, lb_patch_width, len(im_test))
print('Patch center points generated:', len(patch_cpt_test))

   
output = []
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model_save/rd_model-506.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_save/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_pred = graph.get_tensor_by_name("y_pred:0")

    acc = graph.get_tensor_by_name("acc:0")

    num_patch_test = len(patch_cpt_test)
    num_iterations = int(num_patch_test / batch_size_test + 0.5)

    print("Number of iterations: ", num_iterations)

    acc_test = []

    for it in range(num_iterations):
        #print("hou hd")
        #  extract image and label patches in current batch for test data
        im_patch_batch_test, lb_patch_batch_test = dp.data_batch(im_test,
                                                                 lb_test,
                                                                 patch_cpt_test,
                                                                 im_patch_height,
                                                                 im_patch_width,
                                                                 lb_patch_height,
                                                                 lb_patch_width,
                                                                 batch_size_test,
                                                                 it)
        # patch normalization
        im_patch_batch_test = [dp.image_normalize(im) for im in im_patch_batch_test]

        # feed data
        Feed_Dict_Test = {x: im_patch_batch_test, y_true: lb_patch_batch_test}

        _acc_test_ = sess.run(acc, feed_dict=Feed_Dict_Test)
        acc_test.append(_acc_test_)

        # store the prediction for current batch
        output.append(sess.run(y_pred, feed_dict=Feed_Dict_Test))

        if it % 10 == 0:
            msg = "Iteration: {0}...Test acc: {1:>6.1%}"
            print(msg.format(it, _acc_test_))

output_label = dp.pred_mosaic(output, patch_cpt_test)
print(np.shape(output_label))

plt.imshow(output_label)
plt.show()

# write output_label to image file
imsave('test_pred.tif', output_label)



