'''
 VALIDATION CODE SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
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

def show_trajs_test(traj, map, is_gt):

    if (is_gt == True):
        color = (255, 255, 255)
    else:
        color = (0, 0, 255)

    # ground-truth traj
    for j in range(1, traj.shape[0]):

        cur_gt_x = int(traj[j, 0])
        cur_gt_y = int(traj[j, 1])
        pre_gt_x = int(traj[j-1, 0])
        pre_gt_y = int(traj[j-1, 1])

        if (j < 8):
            cv2.line(map, (pre_gt_x, pre_gt_y), (cur_gt_x, cur_gt_y), (255, 0, 0), 2)
        else:
            cv2.line(map, (pre_gt_x, pre_gt_y), (cur_gt_x, cur_gt_y), color, 2)

    return map

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_num', type=int, default=0, help='target dataset number')
    parser.add_argument('--exp_id', type=int, default=0, help='experiment id')
    parser.add_argument('--gpu_num', type=int, default=0, help='target gpu')

    input_args = parser.parse_args()
    test(input_args)

# ------------------------------------------------------
# load saved network and parameters

def test(input_args):

    path = './save_' + str(input_args.dataset_num) + '_' + str(input_args.exp_id)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        try:
            saved_args = pickle.load(f)
            print('>> cpkl file was created under python 3.x')
        except ValueError:
            saved_args = pickle.load(f, encoding='latin1')
            print('>> cpkl file was created under python 2.x')


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

    ADE = []
    FDE = []
    traj = []
    EST_traj = []
    Pid_list = []
    Data_list = []
    cnt = 0

    printProgressBar(0, data_loader.num_test_scenes - 1, prefix='Progress:', suffix='Complete', length=50, epoch=0)
    for b in range(data_loader.num_test_scenes):

        # data load
        xo, xp, xoo, xpo, did, pid = data_loader.next_batch_test([b])
        mo = make_map_batch(xo, did, data_loader.map, saved_args.map_size)

        # prediction
        est_offset = np.squeeze(model.sample(sess, xoo, mo, init_state))  # modification

        # reconstrunction (est traj)
        est_offset_recon = np.concatenate([xoo[0].reshape(saved_args.obs_length, 2), est_offset], axis=0)
        est_offset_recon[0, :] = xo[0][0, :].reshape(1, 2)
        est_traj_recon = np.cumsum(est_offset_recon, axis=0)
        EST_traj.append(est_traj_recon)

        # reconstruction (original)
        x_recon = np.concatenate([xo[0].reshape(obs_len, 2), xp[0].reshape(pred_len, 2)], axis=0)
        traj.append(x_recon)

        if (True):

            map_show = np.copy(data_loader.map[int(did[0][0])])
            map_show = show_trajs_test(x_recon, map_show, True)
            map_show = show_trajs_test(est_traj_recon, map_show, False)

            cv2.imshow('', map_show.astype('uint8'))
            cv2.waitKey(0)


        # calculate error
        err_vector = (est_traj_recon - x_recon)[obs_len:] / 0.25
        displacement_error = np.sqrt(err_vector[:, 0] ** 2 + err_vector[:, 1] ** 2)

        ADE.append(displacement_error)
        FDE.append(displacement_error[pred_len-1])
        cnt += 1

        Pid_list.append(pid)
        Data_list.append(did)

        printProgressBar(b, data_loader.num_test_scenes - 1, prefix='Progress:', suffix='Complete', length=50, epoch=0)

    # save
    save_dir = './test_result_irl_%d.cpkl' % input_args.dataset_num
    f = open(save_dir, "wb")
    pickle.dump([traj, EST_traj, Data_list, Pid_list, data_loader.map, ADE, FDE], f, protocol=2)
    f.close()

if __name__ == '__main__':
    main()