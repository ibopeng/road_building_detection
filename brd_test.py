"""
This script is designed for testing the model trained by rd_train.py

Created on Sun Feb 16 2018
@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import brd_data_proc as dp
import tensorflow as tf
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
num_rawIm_test = 0
# batch size
batch_size_test = int((im_width - im_patch_width) / lb_patch_width + 1)

"""Data Preprocessing: Normalization"""
# load raw images and labels for testing
im_test, lb_test = dp.load_data('./mass_buildings_roads/test', num_rawIm_test)
print('Number of test images loaded: ', num_rawIm_test)

# change labels data type to int32 and set 255 to be 1 s
lb_test = np.array(lb_test)
print('Label data type changed.')

# compute the coordinates of patch center points for one single test image
# note that here 'patch_cpt_test' involves only one image, which is different from that for training and validation
patch_cpt_test = dp.patch_center_point(im_height,
                                       im_width,
                                       im_patch_height,
                                       im_patch_width,
                                       lb_patch_height,
                                       lb_patch_width,
                                       1)
patch_cpt_test = dp.patch_clean(im_test, patch_cpt_test, im_patch_height, im_patch_width)
# number of patches in one single test image
num_patch_test = len(patch_cpt_test)



with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model_save/rd_model.ckpt-506.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_save/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_pred = graph.get_tensor_by_name("y_pred:0")

    acc = graph.get_tensor_by_name("acc:0")


    # average accuracy for all test images
    acc_test_avg = []
    # loop over test set for prediction one by one
    for k in range(len(im_test)):
        print('Predition for test image #{0}'.format(k))
        # number of iterations for this single image
        num_iterations = int(num_patch_test / batch_size_test + 0.5)

        # accuracy for each batch after each iteration
        acc_test_batch = []
        # prediction for each batch
        prediction_batch = []

        for it in range(num_iterations):
            # extract image and label patches in current batch for test data
            # note that the first 2 arguments should be lists
            im_patch_batch_test, lb_patch_batch_test = dp.data_batch([im_test[k]],
                                                                     [lb_test[k]],
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

            # accuracy for this batch
            acc_test_batch.append(sess.run(acc, feed_dict=Feed_Dict_Test))

            # store the prediction for current batch
            prediction_batch.append(sess.run(y_pred, feed_dict=Feed_Dict_Test))

            if it % 10 == 0:
                msg = "Test #{0}...Iteration #{1}...Batch acc: {2:>6.1%}"
                print(msg.format(k, it, acc_test_batch[it]))

        # compute the average accuracy for current test image
        acc_test_avg.append(np.average(np.array(acc_test_batch)))
        print('Test #{0}...Avg Acc: {1:>6.1%}'.format(k, acc_test_avg[k]))

        # mosaiking all batches into one entire image
        im_test_prediction = dp.pred_mosaic(prediction_batch, patch_cpt_test)
        # write output_label to image file
        imsave('./test_prediction/test_pred_{0}.tif'.format(k), im_test_prediction)

    # save the prediction accuracy for all test images
    np.savetxt('./test_prediction/testset_accuracy.txt', acc_test_avg)

#plt.imshow(im_test_prediction)
#plt.show()





