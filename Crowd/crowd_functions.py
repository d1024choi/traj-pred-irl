'''
 FUNCTIONS SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''


import numpy as np
import math
import tensorflow as tf
import sys
import random
import pickle
import os
import copy
#import matplotlib.pyplot as plt
#import cv2


def _max(a,b):
    if (a>b):
        return a
    else:
        return b

def _min(a,b):
    if (a<b):
        return a
    else:
        return b

def moving_average(traj, fsize=3):

    '''
    traj[i, 0] = dataset id
    traj[i, 1] = object id
    traj[i, 2~3] = target pos
    traj[i, 4~63] = neighbor pos

    '''

    seq_length = traj.shape[0]
    processed_traj = np.copy(traj)

    fsize_h = int(fsize/2)

    for i in range(seq_length):
        if (i > fsize_h-1 and i < seq_length-fsize_h):
            processed_traj[i, 2] = np.mean(traj[i-fsize_h:i+fsize_h+1, 2])
            processed_traj[i, 3] = np.mean(traj[i-fsize_h:i+fsize_h+1, 3])

    return processed_traj

def rotate_around_point(xy, degree, origin=(0, 0)):

    radians = math.radians(degree)
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def weight_variable(shape, stddev=0.01, name=None):

    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name=name, initializer=initial)

def bias_variable(shape, init=0.0, name=None):

    initial = tf.constant(init, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name=name, initializer=initial)

def conv_weight_variable(shape, name=None):

    if len(shape) < 4:
        stddev_xavier = math.sqrt(3.0 / (shape[0] + shape[1]))
    else:
        stddev_xavier = math.sqrt(3.0 / ((shape[0]*shape[1]*shape[2]) + (shape[0]*shape[1]*shape[3])))

    initial = tf.truncated_normal(shape, stddev=stddev_xavier)

    return tf.get_variable(initializer=initial, name=name)

def conv_bias_variable(shape, init, name=None):

    initial = tf.constant(init, shape=shape)
    return tf.get_variable(initializer=initial, name=name)

def initialize_conv_filter(shape, name=None):

    W = conv_weight_variable(shape=shape, name=name+'w')
    b = conv_bias_variable(shape=[shape[3]], init=0.0, name=name+'b')

    return W, b

def conv2d_strided_relu(x, W, b, strides, padding):
    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)

    return tf.nn.relu(tf.nn.bias_add(conv, b))

def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="VALID")

def shallow_convnet(input, w1, b1, w2, b2, w3, b3):

    conv1 = conv2d_strided_relu(input, w1, b1, strides=[1, 2, 2, 1], padding='VALID')
    conv2 = conv2d_strided_relu(conv1, w2, b2, strides=[1, 2, 2, 1], padding='VALID')
    conv3 = conv2d_strided_relu(conv2, w3, b3, strides=[1, 2, 2, 1], padding='VALID')
    output = tf.reshape(conv3, [-1, conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2] * conv3.get_shape().as_list()[3]])

    return output

def calculate_reward(fwr, fbr, fc_in, cur_in):
    '''
    :param conv: (1 x conv_flat_size)
    :param cur_in: (1 x self.input_dim)
    :param next_in: (1 x self.input_dim)
    :return:
    '''

    # state_vec = tf.concat([fc_in, cur_in, next_in], axis=1)
    state_vec = tf.concat([fc_in, cur_in], axis=1)
    reward = tf.nn.sigmoid(tf.nn.xw_plus_b(state_vec, fwr, fbr))

    return reward

def measure_accuracy_endpoint(true_traj, est_traj, pred_length):

    seq_length = true_traj.shape[0]
    obs_length = seq_length - pred_length
    error_traj = true_traj - est_traj

    return error_traj[obs_length:seq_length, 0:2]


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>', epoch=10):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(length * iteration / float(total)))
    bar = '>' * filled_length + '-' * (length - filled_length)

    sys.stdout.write('\r [Epoch %02d] %s |%s| %s%s %s' % (epoch, prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
