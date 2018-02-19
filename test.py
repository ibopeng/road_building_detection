"""
This script is designed for testing the model trained by main.py

Created on Sun Feb 16 2018
@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import dataProc as dp
import tensorflow as tf
import designCNN as dc
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
num_rawIm_test = 1
num_rawIm_val_inf = 15
# batch size
batch_size = int((im_width - im_patch_width) / lb_patch_width + 1)

"""Data Preprocessing: Normalization"""
# load raw images and labels for training and validation
im_raw_test, lb_raw_test = dp.load_data('../mass_data_road/data_sub/test', num_rawIm_test)
print('raw data loaded successful')
print('Number of images loaded: ', num_rawIm_test)
# normalization of raw images
im_norm_test = [dp.image_normaliztion(im_raw_test[i]) for i in range(len(im_raw_test))]
print('Image normalization successful.')
# change labels data type to int32 and set 255 to be 1 s
# training data
lb_norm_test = dp.np.array(lb_raw_test)
lb_norm_test = lb_norm_test.astype(dp.np.int32)
lb_norm_test = [lb_norm_test[i] / 255 for i in range(len(lb_norm_test))]
print('Label data type changed successful.')

# compute the coordinates of patch center points
patch_cpt_test = dp.patch_center_point(im_width, im_height, im_patch_width, im_patch_height, lb_patch_width, lb_patch_height, len(im_norm_test))
print('patch center points generated.', len(patch_cpt_test))

   
output = []
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./Model_Save/road_detection_model-100.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./Model_Save/'))

	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name("x:0")
	y_true = graph.get_tensor_by_name("y_true:0")
	y_pred = graph.get_tensor_by_name("y_pred:0")

	acc = graph.get_tensor_by_name("acc:0")

	sz_imtrain_test = len(patch_cpt_test)
	num_iterations = int(sz_imtrain_test / batch_size + 0.5)

	print("Number of iterations: ", num_iterations)
	#num_iterations = 200

	acc_test = []
	

	for it in range(num_iterations):
		#print("hou hd")
		# extract image and label patches in current batch for training data
		im_patch_batch_test, lb_patch_batch_test = dp.data_batch(im_norm_test, lb_norm_test, patch_cpt_test, im_patch_width, im_patch_height, lb_patch_width, lb_patch_height, batch_size, it)

		Feed_Dict_Test = {x: im_patch_batch_test, y_true: lb_patch_batch_test}
		acc_test.append(sess.run(acc, feed_dict = Feed_Dict_Test))

		# store the prediction for current batch
		output.append(sess.run(y_pred, feed_dict = Feed_Dict_Test))


		if it % 10 == 0:
			msg = "Iteration: {0}...Test acc: {1:>6.1%}"
			print(msg.format(it, acc_test[it]))
		
output_label = dp.image_mosaic(output, patch_cpt_test)
print(dp.np.shape(output_label))

plt.imshow(output_label)
plt.show()



