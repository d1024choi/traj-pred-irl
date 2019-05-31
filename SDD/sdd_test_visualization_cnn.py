'''
 VALIDATION CODE SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 VERSION : 0.1 (2019-01-09)
 DESCRIPTION : ...
'''


from sdd_model import Model
from sdd_utils import DataLoader
from sdd_functions import *
import pickle
import os
import argparse
import cv2
#import matplotlib.pyplot as plt

def sigmoid(img):

    # sig = 1 / (1 + np.exp(-9.0 * (img - 0.5)))
    img = img -np.min(img[:])
    sig = img / np.max(img[:])

    return sig

def show_trajs_test(gt_traj, est_traj, map):

    # ground-truth traj
    #for j in range(1, gt_traj.shape[0]):
    for j in range(1, gt_traj.shape[0]):

        cur_gt_x = int(gt_traj[j, 0])
        cur_gt_y = int(gt_traj[j, 1])
        pre_gt_x = int(gt_traj[j-1, 0])
        pre_gt_y = int(gt_traj[j-1, 1])

        if (j < 8):
            cv2.line(map, (pre_gt_x, pre_gt_y), (cur_gt_x, cur_gt_y), (0, 0, 0), 2)
        else:
            cv2.line(map, (pre_gt_x, pre_gt_y), (cur_gt_x, cur_gt_y), (255, 255, 255), 2)

    for j in range(1, gt_traj.shape[0]):

        cur_gt_x = int(est_traj[j, 0])
        cur_gt_y = int(est_traj[j, 1])
        pre_gt_x = int(est_traj[j-1, 0])
        pre_gt_y = int(est_traj[j-1, 1])

        if (j < 8):
            donothing = 0
        else:
            cv2.line(map, (pre_gt_x, pre_gt_y), (cur_gt_x, cur_gt_y), (0, 0, 255), 2)

    return map

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_num', type=int, default=0,
                        help='target dataset number')
    parser.add_argument('--exp_id', type=int, default=26,
                        help='experiment id')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='target gpu')

    input_args = parser.parse_args()
    test(input_args)

# ------------------------------------------------------
# load saved network and parameters

def test(input_args):

    path = './save_' + str(input_args.dataset_num) + '_' + str(input_args.exp_id)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
        #saved_args = pickle.load(f, encoding='latin1')

    if (input_args.gpu_num == 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif (input_args.gpu_num == 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif (input_args.gpu_num == 2):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    elif (input_args.gpu_num == 3):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    obs_len = saved_args.obs_length
    pred_len = saved_args.pred_length

    # define model structure
    model = Model(saved_args, True)

    # load trained weights
    ckpt = tf.train.get_checkpoint_state(path)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)
    print(">> loaded model: ", ckpt.model_checkpoint_path)


    # ------------------------------------------------------
    # variable definition for validation

    init_state = np.zeros(shape=(1, 1, 2*saved_args.rnn_size))

    # load validation data
    data_loader = DataLoader(saved_args)

    rand_list = [x for x in range(data_loader.num_test_scenes)]
    random.shuffle(rand_list)

    for b in range(data_loader.num_test_scenes):

        if (b == 37300):
        #if (True and b % 100 == 0):


            # data load
            xo, xp, xoo, xpo, did, pid = data_loader.next_batch_test([b])
            #xo, xp, xoo, xpo, did, pid = data_loader.next_batch_test([rand_list[b]])
            mo = make_map_batch(xo, did, data_loader.map, saved_args.map_size)

            # prediction
            est_offset, conv_out = np.squeeze(model.sample(sess, xoo, mo, init_state))  # modification

            # reconstrunction (est traj)
            est_offset_recon = np.concatenate([xoo[0].reshape(saved_args.obs_length, 2), est_offset], axis=0)
            est_offset_recon[0, :] = xo[0][0, :].reshape(1, 2)
            est_traj_recon = np.cumsum(est_offset_recon, axis=0)

            # reconstruction (original)
            x_recon = np.concatenate([xo[0].reshape(obs_len, 2), xp[0].reshape(pred_len, 2)], axis=0)

            # show cnn results
            map_show = np.copy(data_loader.map[int(did[0][0])])
            row = map_show.shape[0]
            col = map_show.shape[1]

            image = mo[0][7]
            #conv = np.sum(np.abs(np.array(conv_out)), axis=3)
            conv = np.sum(conv_out, axis=3)
            conv = cv2.resize(conv[7], (96, 96), interpolation=cv2.INTER_CUBIC)
            conv = sigmoid(conv)
            conv = np.expand_dims(conv, axis=2) * 255.0
            conv[:2, :] = 0
            conv[-2:, :] = 0
            conv[:, -2:] = 0
            conv[:, :2] = 0
            conv_3d = np.concatenate([conv, conv, conv], axis=2)
            #conv_3d = np.concatenate([conv, np.zeros_like(conv), np.zeros_like(conv)], axis=2)


            #final = np.concatenate([image, conv_3d], axis=1)

            x_center = int(x_recon[7, 0].astype('int32'))
            y_center = int(x_recon[7, 1].astype('int32'))

            # 2) overlap images on the entire image
            if (y_center - 48 < 0 or y_center + 48 > row-1 or x_center - 48 < 0 or x_center + 48 > col -1):
                continue
            map_show[y_center - 48: y_center + 48, x_center - 48:x_center + 48, :] = conv_3d
            #roi = np.copy(map_show[y_center - 48: y_center + 48, x_center - 48:x_center + 48, :])
            #map_show[y_center - 48: y_center + 48, x_center - 48:x_center + 48, :] = (0.2*roi + 0.8*conv_3d)
            map_show = show_trajs_test(x_recon, est_traj_recon, map_show)

            file_name_cnn = 'IRL_D%d_S%d.bmp' % (input_args.dataset_num, b)
            cv2.imwrite(file_name_cnn, map_show.astype('uint8'))

            cv2.imshow('', map_show.astype('uint8'))
            cv2.waitKey(0)






if __name__ == '__main__':
    main()