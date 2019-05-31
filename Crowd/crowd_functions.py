'''
 FUNCTIONS SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 VERSION : 0.9 (2018-08-14)
 DESCRIPTION : ...
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

def random_flip(x, y):

    '''
    (confirmed) randomly flip data in the direction of x and y axis
    '''

    flipud = 0
    if (np.random.rand(1) < 0.5):
        flipud = 1
        x[:, 0] = -1.0 * x[:, 0]
        y[y[:, :, 0] > -1000, 0] = -1.0 * y[y[:, :, 0] > -1000, 0]

    fliplr = 0
    if (np.random.rand(1) < 0.5):
        fliplr = 1
        x[:, 1] = -1.0 * x[:, 1]
        y[y[:, :, 1] > -1000, 1] = -1.0 * y[y[:, :, 1] > -1000, 1]

    return x, y, flipud, fliplr

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

def random_rotate(tpos, npos):

    '''
    (confirmed) randomly rotate trajectory
     - must be used non-processed trajectory

    tpos = seq_len x 2
    npos = seq_len x 30 x 2
    '''

    tpos_rot = np.copy(tpos)
    npos_rot = np.copy(npos)
    origin = (tpos[0, 0], tpos[0, 1])
    degree = random.randint(1, 359)

    if (np.random.rand(1) < 0.5):
        for i in range(0, tpos.shape[0]):

            # rotate target pos
            rx, ry = rotate_around_point((tpos[i, 0], tpos[i, 1]), degree, origin)
            tpos_rot[i, 0] = rx
            tpos_rot[i, 1] = ry

            # rotate neighbors
            for j in range(30):
                if (npos[i, j, 0] == -1000):
                    continue
                else:
                    rx, ry = rotate_around_point((npos[i, j, 0], npos[i, j, 1]), degree, origin)
                    npos_rot[i, j, 0] = rx
                    npos_rot[i, j, 1] = ry

    return tpos_rot, npos_rot

def reform_neighbor_traj(xn):

    '''
    :param xn: seq_length-1 x 50 x 3
    '''

    nList = np.unique(xn[:, :, 2])
    nTrajs = []

    for _, nped_id in enumerate(nList):
        nTraj = []
        for i in range(xn.shape[0]):
            for j in range(xn.shape[1]):
                if (xn[i, j, 2] == nped_id and xn[i, j, 0] > -1000.0):
                    nTraj.append(xn[i, j, 0:2])
        nTrajs.append(np.squeeze(np.array(nTraj)))


    return nTrajs

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
    # pool1 = max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv2 = conv2d_strided_relu(conv1, w2, b2, strides=[1, 2, 2, 1], padding='VALID')
    # pool2 = max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv3 = conv2d_strided_relu(conv2, w3, b3, strides=[1, 2, 2, 1], padding='VALID')
    # pool3 = max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

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

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-20

    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    # s1s2 = tf.multiply(s1, s2)
    s1s2 = tf.add(tf.multiply(s1, s2), epsilon)

    z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
        2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)

    negRho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * negRho))

    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))

    result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):

    result0 = tf_2d_normal(
        x1_data,
        x2_data,
        z_mu1,
        z_mu2,
        z_sigma1,
        z_sigma2,
        z_corr)

    # implementing eq # 26 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-20
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    # at the beginning, some errors are exactly zero.
    result = -tf.log(tf.maximum(result1, epsilon))

    return tf.reduce_sum(result)

def get_mixture_coef(output):
    # returns the tf slices containing mdn dist params
    # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
    z = output
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
        axis=1, num_or_size_splits=6, value=z[:, 0:])

    # process output z's into MDN paramters

    # softmax all the pi's:
    max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
    z_pi = tf.subtract(z_pi, max_pi)
    z_pi = tf.exp(z_pi)
    normalize_pi = tf.reciprocal(
        tf.reduce_sum(z_pi, 1, keep_dims=True))
    z_pi = tf.multiply(normalize_pi, z_pi)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def preprocessing(traj, data_scale):

    '''
    :param traj: (seq_length+1, input_dim)
    :return:
    '''

    processed_data = np.copy(traj)
    prev_x = 0
    prev_y = 0

    for t in range(traj.shape[0]):
        processed_data[t, 0] = traj[t, 0] - prev_x
        processed_data[t, 1] = traj[t, 1] - prev_y

        prev_x = traj[t, 0]
        prev_y = traj[t, 1]

    processed_data[:, 0:2] /= data_scale

    return processed_data

def postprocessing(traj, data_scale):

    traj_ext = np.copy(traj)
    traj_ext *= data_scale

    for t in range(1, traj_ext.shape[0]):
        traj_ext[t, 0] += traj_ext[t - 1, 0]
        traj_ext[t, 1] += traj_ext[t - 1, 1]

    return traj_ext

def measure_accuracy_overall(true_traj, est_traj, pred_length):

    seq_length = true_traj.shape[0]
    obs_length = seq_length - pred_length

    x_diff, y_diff = 0, 0
    for i in range(obs_length, seq_length):
        x_diff += abs(true_traj[i, 0] - est_traj[i, 0])
        y_diff += abs(true_traj[i, 1] - est_traj[i, 1])

    return (x_diff/pred_length), (y_diff/pred_length)

def measure_accuracy_endpoint(true_traj, est_traj, pred_length):

    seq_length = true_traj.shape[0]
    obs_length = seq_length - pred_length
    error_traj = true_traj - est_traj

    return error_traj[obs_length:seq_length, 0:2]

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
