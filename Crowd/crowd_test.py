'''
 VALIDATION CODE SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''


from crowd_model import Model
from crowd_utils import DataLoader
from crowd_functions import *
from crowd_occupancy import *
import pickle
import os
import argparse
#import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_num', type=int, default=3, help='target dataset number')
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

    # debug --
    saved_args.max_num_peds = 80
    saved_args.is_pretrain = 0

    obs_len = saved_args.obs_length
    pred_len = saved_args.pred_length
    MNP = saved_args.max_num_peds

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

    init_state = np.zeros(shape=(1, saved_args.max_num_peds, 2*saved_args.rnn_size))

    # load validation data
    data_loader = DataLoader(saved_args)

    ADE = 0.0
    FDE = 0.0
    cnt = 0
    for b in range(data_loader.num_test_scenes):

        nps, xo, xp, xoo, xpo, xm, pid = data_loader.next_batch_test([b])
        ogm = rectangular_occupancy_map(xo, MNP, nps, obs_len, saved_args.social_range, saved_args.grid_size)

        # run prediction
        est_offset = np.squeeze(model.sample(sess, xoo, init_state, ogm)) # modification

        # reconstrunction (est traj)
        est_offset_recon = np.concatenate([xoo[0].reshape(MNP, saved_args.obs_length, 2), est_offset], axis=1)
        est_offset_recon[:, 0, :] = xo[0][:, 0, :].reshape(MNP, 2)
        est_traj_recon = np.cumsum(est_offset_recon, axis=1)

        # reconstruction (original)
        x_recon = np.concatenate([xo[0].reshape(MNP, obs_len, 2), xp[0].reshape(MNP, pred_len, 2)], axis=1)

        # calculate error
        for j in range(nps[0]):
            err_vector = measure_accuracy_endpoint(x_recon[j], est_traj_recon[j], pred_len)
            displacement_error = np.sqrt(err_vector[:, 0] ** 2 + err_vector[:, 1] ** 2)

            ADE += np.mean(displacement_error)
            FDE += displacement_error[pred_len-1]
            cnt += 1

    ADE = ADE / float(cnt)
    FDE = FDE / float(cnt)
    print('--------- dataset number : %d ---------' % saved_args.dataset_num)
    print('average displacement error : %.4f' % ADE)
    print('final displacement error : %.4f' % FDE)


    file_name_txt = 'test_result_' + str(input_args.dataset_num) + '.txt'
    file = open(os.path.join(path, file_name_txt), "w")
    file.write('ADE: ' + str(ADE) + '____')
    file.write('FDE: ' + str(FDE) + '____')
    file.close()

if __name__ == '__main__':
    main()
