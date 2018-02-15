#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:56:41 2018

@ Author: Bo Peng
@ University of Wisconsin - Madison
@ Project: Road Extraction
"""

import tensorflow as tf

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