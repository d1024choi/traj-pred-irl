'''
 FUNCTIONS SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 VERSION : 2.1 (2019-01-23)
 DESCRIPTION : ...
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

def moving_average(traj, fsize=3):

    '''
    traj[i, 0] = dataset id
    traj[i, 1] = object id
    traj[i, 2~3] = target pose
    traj[i, 4~63] = neighbor pose

    '''

    seq_length = traj.shape[0]
    processed_traj = np.copy(traj)

    fsize_h = int(fsize/2)

    for i in range(seq_length):
        if (i > fsize_h-1 and i < seq_length-fsize_h):
            processed_traj[i, 2] = np.mean(traj[i-fsize_h:i+fsize_h+1, 2])
            processed_traj[i, 3] = np.mean(traj[i-fsize_h:i+fsize_h+1, 3])

    return processed_traj

def getSocialMatrix(socialVec, target_pose, neighbor_pose, socialRange, grid_size):

    '''
    :param socialVec: (num_grid, num_grid)
    :param target_pose: (seq_length, 2)
    :param neighbor_pose: (seq_length, 60)
    :param socialRange:
    :param grid_size:
    '''

    num_grid = int(socialRange / grid_size)

    delta_x = neighbor_pose[0, 0] - target_pose[0, 0] + (socialRange/2)
    delta_y = neighbor_pose[0, 1] - target_pose[0, 1] + (socialRange/2)

    grid_idx_x = int(delta_x / grid_size)
    grid_idx_y = (num_grid - 1) - int(delta_y / grid_size)

    # debug
    if (grid_idx_x < 0 or grid_idx_x > (num_grid-1) or grid_idx_y < 0 or grid_idx_y > (num_grid-1)):
        donothing = 0
    else:
        # socialVec[grid_idx_x, grid_idx_y] = 1
        # image-x-axis corresponds to array-column
        socialVec[grid_idx_y, grid_idx_x] = 1

    return socialVec

def random_flip(x, map):

    '''
    (confirmed) randomly flip data in the direction of x and y axis
    '''

    flipud = 0
    if (np.random.rand(1) < 0.5):
        flipud = 1
        x[:, 0] = -1.0 * x[:, 0]
        for i in range(len(map)):
            map[i] = np.flipud(map[i])

    fliplr = 0
    if (np.random.rand(1) < 0.5):
        fliplr = 1
        x[:, 1] = -1.0 * x[:, 1]
        for i in range(len(map)):
            map[i] = np.fliplr(map[i])

    return x, map, flipud, fliplr

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

def random_rotate(tpose, npose):

    '''
    (confirmed) randomly rotate trajectory
     - must be used non-processed trajectory

    tpose = seq_len x 2
    npose = seq_len x 30 x 2
    '''

    tpose_rot = np.copy(tpose)
    npose_rot = np.copy(npose)
    origin = (tpose[0, 0], tpose[0, 1])
    degree = random.randint(1, 359)

    if (np.random.rand(1) < 0.5):
        for i in range(0, tpose.shape[0]):

            # rotate target pose
            rx, ry = rotate_around_point((tpose[i, 0], tpose[i, 1]), degree, origin)
            tpose_rot[i, 0] = rx
            tpose_rot[i, 1] = ry

            # rotate neighbors
            for j in range(30):
                if (npose[i, j, 0] == -1000):
                    continue
                else:
                    rx, ry = rotate_around_point((npose[i, j, 0], npose[i, j, 1]), degree, origin)
                    npose_rot[i, j, 0] = rx
                    npose_rot[i, j, 1] = ry

    return tpose_rot, npose_rot

def map_roi_extract(x, y, map, width, fidx):

    size_row = map.shape[0]
    size_col = map.shape[1]

    x_center = int(x.astype('int32')) # width
    y_center = int(y.astype('int32')) # height

    # improved 180523
    if (x_center-width < 0 or x_center+width-1 > size_col-1 or y_center-width < 0 or y_center+width-1 > size_row-1):


        #x_center = int(x.astype('int32')) + 2*width
        #y_center = int(y.astype('int32')) + 2*width

        #map_pad = np.zeros(shape=(size_row + 4*width, size_col + 4*width, 3))
        #map_pad[2*width:size_row+2*width, 2*width:2*width+size_col, :] = np.copy(map)

        #part_map = np.copy(map_pad[y_center - width:y_center + width, x_center - width:x_center + width, :])

        #size_row_p = part_map.shape[0]
        #size_col_p = part_map.shape[1]

        #if (size_row_p != width*2 or size_col_p != width*2):
        #    breakpoint = 0

        #    # part_map = np.zeros(shape=(2*width, 2*width, 3))
        #    cv2.imshow('', map_pad.astype('uint8'))
        #    cv2.waitKey(0)

        part_map = np.zeros(shape=(2 * width, 2 * width, 3))
    else:
        part_map = np.copy(map[y_center - width:y_center + width, x_center - width:x_center + width, :])


    if (fidx == 0):
        part_map = np.zeros(shape=(2 * width, 2 * width, 3))

    # debug ---
    #part_map[:2, :, :] = 255
    #part_map[:, :2, :] = 255
    #part_map[-4:-2, :, :] = 255
    #part_map[:, -4:-2, :] = 255

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
    # pool1 = max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv2 = conv2d_strided_relu(conv1, w2, b2, strides=[1, 2, 2, 1], padding='VALID')
    # pool2 = max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv3 = conv2d_strided_relu(conv2, w3, b3, strides=[1, 2, 2, 1], padding='VALID')
    # pool3 = max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    output = tf.reshape(conv3, [-1, conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2] * conv3.get_shape().as_list()[3]])

    return output, conv3

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

def plot_trajectories(x, est_traj_post, pred_length):

    plt.plot(x[:, 0], x[:, 1], 'bo-', label='true traj.')
    plt.plot(est_traj_post[x.shape[0] - pred_length - 1:, 0], est_traj_post[x.shape[0] - pred_length - 1:, 1], 'y+-.',
             label='est. traj.')
    plt.plot(est_traj_post[0:x.shape[0] - pred_length, 0], est_traj_post[0:x.shape[0] - pred_length, 1], 'y+-',
             label='input for est.')
    plt.legend()
    min_x = np.min(x[:, 0])
    min_y = np.min(x[:, 1])

    max_x = np.max(x[:, 0])
    max_y = np.max(x[:, 1])

    diff_x = abs(np.min(x[:, 0]) - np.max(x[:, 0]))
    diff_y = abs(np.min(x[:, 1]) - np.max(x[:, 1]))
    width = abs(diff_x - diff_y) / 2

    mr = 0.5
    if (diff_x < 2 and diff_y < 2):
        mr = _max(2-diff_x, 2-diff_y) + 0.5

    if (diff_x > diff_y):
        plt.axis([min_x-mr, max_x+mr, min_y-width-mr, max_y+width+mr])
    else:
        plt.axis([min_x-width-mr, max_x+width+mr, min_y-mr, max_y+mr])
    plt.show()

def plot_trajectories_on_map(x_gt, x_est, pred_length, map, x_max, y_max, scale):

    seq_length = x_gt.shape[0]
    obs_length = seq_length - pred_length

    x_gt_max = np.max(x_gt[:, 0])
    x_gt_min = np.min(x_gt[:, 0])

    y_gt_max = np.max(x_gt[:, 1])
    y_gt_min = np.min(x_gt[:, 1])

    x_axis_width = abs(x_gt_max - x_gt_min)
    y_axis_width = abs(y_gt_max - y_gt_min)
    map_size = scale*(3.0*_max(x_axis_width, y_axis_width) + 3.0)
    map_size = _max(_min(map_size, 400), 100)

    x = x_gt[obs_length - 1, 0]
    y = x_gt[obs_length - 1, 1]
    map_roi = np.copy(map_roi_extract(map, x, y, x_max, y_max, scale, int(map_size/2)))
    map_roi_copy = np.copy(map_roi)
    map_row_cnt = map_roi.shape[0] / 2
    map_col_cnt = map_roi.shape[1] / 2

    pose_start_x = x
    pose_start_y = y

    for i in range(seq_length):

        pose_x = int(scale * (x_gt[i, 0] - pose_start_x) + map_row_cnt)
        pose_y = int(scale * (x_gt[i, 1] - pose_start_y) + map_col_cnt)

        pose_x = _min(_max(pose_x, 0), map_roi.shape[0] - 1)
        pose_y = _min(_max(pose_y, 0), map_roi.shape[1] - 1)

        map_roi[pose_x, pose_y, 0] = 255
        map_roi[pose_x, pose_y, 1] = 255
        map_roi[pose_x, pose_y, 2] = 255

        if (i > obs_length-1):
            pose_x = int(scale * (x_est[i, 0] - pose_start_x) + map_row_cnt)
            pose_y = int(scale * (x_est[i, 1] - pose_start_y) + map_col_cnt)

            pose_x = _min(_max(pose_x, 0), map_roi.shape[0] - 1)
            pose_y = _min(_max(pose_y, 0), map_roi.shape[1] - 1)

            map_roi[pose_x, pose_y, 0] = 20
            map_roi[pose_x, pose_y, 1] = 255
            map_roi[pose_x, pose_y, 2] = 57

    #cv2.imshow('test', map_roi)
    #cv2.waitKey(0)

    return map_roi_copy


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
