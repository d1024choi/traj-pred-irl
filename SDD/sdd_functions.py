'''
 FUNCTIONS SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''

import numpy as np
import math
import random
import tensorflow as tf
import sys
#import matplotlib.pyplot as plt
import cv2
import copy

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

def map_roi_extract(x, y, map, width, fidx):

    size_row = map.shape[0]
    size_col = map.shape[1]

    x_center = int(x.astype('int32')) # width
    y_center = int(y.astype('int32')) # height

    # improved 180523
    if (x_center-width < 0 or x_center+width-1 > size_col-1 or y_center-width < 0 or y_center+width-1 > size_row-1):


        ## validation code ----------------------------------
        # x_center = int(x.astype('int32')) + 2*width
        # y_center = int(y.astype('int32')) + 2*width
        # map_pad = np.zeros(shape=(size_row + 4*width, size_col + 4*width, 3))
        # map_pad[2*width:size_row+2*width, 2*width:2*width+size_col, :] = np.copy(map)
        # part_map = np.copy(map_pad[y_center - width:y_center + width, x_center - width:x_center + width, :])
        # size_row_p = part_map.shape[0]
        # size_col_p = part_map.shape[1]
        # if (size_row_p != width*2 or size_col_p != width*2):
        #    cv2.imshow('', map_pad.astype('uint8'))
        #    cv2.waitKey(0)

        part_map = np.zeros(shape=(2 * width, 2 * width, 3))
    else:
        part_map = np.copy(map[y_center - width:y_center + width, x_center - width:x_center + width, :])

    if (fidx == 0):
        part_map = np.zeros(shape=(2 * width, 2 * width, 3))

    return part_map

def make_map_batch(xo_batch, did_batch, maps, target_map_size):


    half_map_size = int(target_map_size / 2)
    target_map = []
    for k in range(len(xo_batch)):

        # current traj
        xo = xo_batch[k]

        # current map
        map = maps[int(did_batch[k][0])]

        # debug ---
        '''
        width = map.shape[1]
        height = map.shape[0]
        map_show = np.zeros(shape=(height + 4*half_map_size, width + 4*half_map_size, 3))
        map_show[2*half_map_size:height+2*half_map_size, 2*half_map_size:2*half_map_size+width, :] = np.copy(map)
        '''

        map_seq = []
        for i in range(xo.shape[0]):
            x = xo[i, 0]
            y = xo[i, 1]

            corr_map = map_roi_extract(x, y, map, half_map_size, i)
            map_seq.append(corr_map)

            # debug ---
            '''
            # # TEST code # #
            x_center = int(x.astype('int32')) + 2*half_map_size
            y_center = int(y.astype('int32')) + 2*half_map_size

            # 2) overlap images on the entire image
            #if (x_center - half_map_size < 0 or x_center + half_map_size - 1 > width - 1 or y_center - half_map_size < 0 or y_center + half_map_size - 1 > height - 1):
            #    continue
            map_show[y_center - half_map_size: y_center + half_map_size, x_center - half_map_size:x_center + half_map_size, :] = corr_map

            # 2) overlap images on the entire image
            color_code = 255
            map_show = cv2.circle(map_show, (x_center, y_center), 3, (0, 0, color_code), -1)

            cv2.imshow('test', map_show.astype('uint8'))
            cv2.waitKey(0)
            '''

        target_map.append(map_seq)

    return target_map

def make_map_batch_for_policy(xo_batch, xoo_batch, xoo_p_batch, did_batch, maps, target_map_size):

    '''
    original sequence :    xo = [x0, x1, x2 x3]
    encoder input :      xoo  = [x0-x0, x1-x0, x2-x1, x3-x2]
                              = [0,     dx1,   dx2,   dx3]
    encoder output :    xoo_p = [dx1^,  dx2^,  dx3^,  dx4^]

    xoo_p_batch_shift[0, dx1^, dx2^, dx3^] = xoo_p_batch[dx1^, dx2^, dx3^]

    question) dx1^ is always zero. Therefore, shifted result is [0, 0, dx2^, dx3^].
              The result before cumsum is now [x0, 0, dx2^, dx3^].
              The cumsum result is now [x0, x0, x0+dx2^, x0+dx2^+dx3^]
              The original cumsum should be [x0, x0+dx1^, x0+dx1^+dx2^, x0+dx1^+dx2^+dx3^]
    '''

    target_map = []
    for k in range(len(xo_batch)):

        # reconstruct est traj
        xo = xo_batch[k]
        xoo = xoo_batch[k]
        xoo_p = xoo_p_batch[k]

        xoo_p_shift = np.zeros_like(xoo_p)
        xoo_p_shift[1:] = xoo_p[:-1]

        xoo_p_shift[0, :] = xo[0, :]
        xoo_p_shift[1, :] = xoo[1, :]
        xoo_p_recon = np.cumsum(xoo_p_shift, axis=0)

        # current map
        map = maps[int(did_batch[k][0])]
        map_seq = []
        for i in range(xo.shape[0]):
            x = xoo_p_recon[i, 0]
            y = xoo_p_recon[i, 1]

            corr_map = map_roi_extract(x, y, map, int(target_map_size / 2), i)
            map_seq.append(corr_map)

        target_map.append(map_seq)

    return target_map

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


def reshape_est_traj(x, batch_size, seq_length):

    '''
    :param np.squeeze(x): (batch_sizexseq_length, 2)
    :return: x_reshape
    '''

    x_np = np.squeeze(np.array(x))
    x_reshape = []

    for i in range(batch_size):
        x_reshape.append(x_np[i*seq_length:(i+1)*seq_length, :])

    return x_reshape

# Print iterations progress
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
