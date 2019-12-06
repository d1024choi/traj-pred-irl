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
#import matplotlib.pyplot as plt

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
    cnt = 0

    printProgressBar(0, data_loader.num_test_scenes - 1, prefix='Progress:', suffix='Complete', length=50, epoch=0)
    for b in range(data_loader.num_test_scenes):

        xo, xp, xoo, xpo, did, pid = data_loader.next_batch_test([b])
        mo = make_map_batch(xo, did, data_loader.map, saved_args.map_size)

        # run prediction
        est_offset = np.squeeze(model.sample(sess, xoo, mo, init_state)) # modification

        # reconstrunction (est traj)
        est_offset_recon = np.concatenate([xoo[0].reshape(saved_args.obs_length, 2), est_offset], axis=0)
        est_offset_recon[0, :] = xo[0][0, :].reshape(1, 2)
        est_traj_recon = np.cumsum(est_offset_recon, axis=0)

        # reconstruction (original)
        x_recon = np.concatenate([xo[0].reshape(obs_len, 2), xp[0].reshape(pred_len, 2)], axis=0)

        # calculate error
        err_vector = (est_traj_recon - x_recon)[obs_len:] / 0.25
        displacement_error = np.sqrt(err_vector[:, 0] ** 2 + err_vector[:, 1] ** 2)

        ADE.append(displacement_error)
        FDE.append(displacement_error[pred_len-1])
        cnt += 1

        printProgressBar(b, data_loader.num_test_scenes - 1, prefix='Progress:', suffix='Complete', length=50, epoch=0)

    print('--------- dataset number : %d ---------' % saved_args.dataset_num)
    print('average displacement error : %.4f' % np.mean(ADE))
    print('final displacement error : %.4f' % np.mean(FDE))

    file_name_txt = 'test_result_' + str(input_args.dataset_num) + '.txt'
    file = open(os.path.join(path, file_name_txt), "w")
    file.write('ADE: ' + str(np.mean(ADE)) + '____')
    file.write('FDE: ' + str(np.mean(FDE)) + '____')
    file.close()

if __name__ == '__main__':
    main()
