'''
 SDD TRAJECTORY PREDICTION TRAINING
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''

import argparse
import os
import pickle
import time

from sdd_model import Model
from sdd_utils import DataLoader
from sdd_functions import *


def print_training_info(args):

    print('-----------------------------------------------------------')
    print('------------ Encoder-Decoder LSTM INFORMATION -------------')
    print('.dataset_num %d, exp id %d' %(args.dataset_num, args.exp_id))
    print('.network structure: LSTM')
    print('   rnn size (%d)' % (args.rnn_size))
    print('   num layers (%d)' % (args.num_layers))
    print('.training setting')
    print('   num epochs (%d)' % args.num_epochs)
    print('   batch size (%d)' % args.batch_size)
    print('   obs_length (%d)' % args.obs_length)
    print('   pred_length (%d)' % args.pred_length)
    print('   map size (%d)' % args.map_size)
    print('   learning rate (%.5f)' % args.learning_rate)
    print('   reg. lambda (%.5f)' % args.lambda_param)
    print('   gamma param (%.5f)' % args.gamma_param)
    print('   grad_clip (%.2f)' % args.grad_clip)
    print('   keep prob (%.2f)' % args.keep_prob)
    print('   load pretrained (%d)' % args.load_pretrained)
    print('   start epoch (%d)' % args.start_epoch)
    print('   data load step (%d)' % args.data_load_step)
    print('------------------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()

    # # network structure : LSTM
    parser.add_argument('--rnn_size', type=int, default=256, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm', help='rnn, gru, or lstm')
    parser.add_argument('--input_dim', type=int, default=2, help='dimension of input vector')

    # # training setting
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='dataset path')
    parser.add_argument('--exp_id', type=int, default=0, help='dataset id')
    '''
    dataset number | random seed
    -----------------------------
           0       |      2
    -----------------------------
           1       |      2
    -----------------------------
           2       |      2
    -----------------------------
           3       |      2
    -----------------------------
           4       |      2                                           
    '''
    parser.add_argument('--dataset_num', type=int, default=0, help='dataset number')
    parser.add_argument('--gpu_num', type=int, default=2, help='gpu device id utilized for training and validation')
    parser.add_argument('--load_pretrained', type=int, default=0, help='want to load pre-trained network?')
    parser.add_argument('--batch_size', type=int, default=4, help='minibatch size')
    parser.add_argument('--obs_length', type=int, default=8, help='observation sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save', help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=1.5, help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lambda_param', type=float, default=0.0001, help='l2-regularization weight')
    parser.add_argument('--gamma_param', type=float, default=0.001, help='IRL regularization weight')
    parser.add_argument('--keep_prob', type=float, default=0.8, help='dropout keep probability')
    parser.add_argument('--patient_thr', type=float, default=100, help='threshold for early stopping')
    parser.add_argument('--data_load_step', type=int, default=3, help='data_load_step for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch value')
    parser.add_argument('--min_avg_loss', type=float, default=100000.0, help='min avg loss')

    # # map info
    parser.add_argument('--map_size', type=int, default=96, help='width of map image')

    args = parser.parse_args()
    train(args)


def train(args):

    # print training information
    print_training_info(args)

    # assign gpu device
    if (args.gpu_num == 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif (args.gpu_num == 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif (args.gpu_num == 2):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    elif (args.gpu_num == 3):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # training data preparation (utils.py)
    data_loader = DataLoader(args)

    # check if there is pre-trained network
    args.model_dir = args.model_dir + '_' + str(args.dataset_num) + '_' + str(args.exp_id)
    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        args.load_pretrained = 0

    # load saved training parameters or save current parameters
    start_epoch = args.start_epoch
    if (args.load_pretrained == 1):
        with open(os.path.join(args.model_dir, 'config.pkl'), 'rb') as f:
            args = pickle.load(f)
            args.load_pretrained = 1
    else:
        with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)

    # network model definition (model.py)
    model = Model(args)

    with tf.Session() as sess:

        # initialize network
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        # load the latest trained parameters
        if (args.load_pretrained == 1):
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(">> loaded model: ", ckpt.model_checkpoint_path)

        best_epoch = 0
        patient = 0
        min_avg_loss = 10000.0
        if (args.load_pretrained == 1):
            min_avg_loss = args.min_avg_loss
        print('>> training and validation start from epoch %d  with min_avg_loss %.4f' % (start_epoch, min_avg_loss))

        # start training and validation !!!!
        for e in range(start_epoch, args.num_epochs):

            state = model.init_state_enc.eval()

            rand_list = [x for x in range(data_loader.num_train_scenes)]
            random.shuffle(rand_list)

            # # Train one epoch -------------------------------------------------------
            printProgressBar(0, data_loader.num_batches - 1, prefix='Progress:', suffix='Complete', length=50, epoch=e)
            start = time.time()

            for b in range(data_loader.num_batches):

                # load mini-batch
                xo, xp, xoo, xpo, did = data_loader.next_batch(rand_list[b * args.batch_size:(b + 1) * args.batch_size])
                mo = make_map_batch(xo, did, data_loader.map, args.map_size)

                # step 1) simulate current policy
                feed = {model.gt_traj_enc: xoo,
                        model.gt_map_enc: mo,
                        model.init_state_enc: state,
                        model.output_keep_prob: 1.0}

                xoo_policy = reshape_est_traj(sess.run([model.predictions_enc], feed), args.batch_size, args.obs_length)
                mo_policy = make_map_batch_for_policy(xo, xoo, xoo_policy, did, data_loader.map, args.map_size)

                # step2) train rnn
                feed = {model.gt_traj_enc: xoo,
                        model.gt_traj_dec: xpo,
                        model.gt_map_enc: mo,
                        model.policy_map_enc: mo_policy,
                        model.init_state_enc: state,
                        model.output_keep_prob: args.keep_prob}
                train_loss, _ = sess.run([model.cost_pos_dec, model.train_op_pose], feed)

                # step3) train reward
                train_loss, _ = sess.run([model.cost_reward, model.train_op_reward], feed)

                printProgressBar(b, data_loader.num_batches - 1, prefix='Progress:', suffix='Complete', length=50, epoch=e)
            end = time.time()

            # # Validation ------------------------------------------------------------
            valid_loss_list = []
            state = model.init_state_enc.eval()
            traj_list = [x for x in range(data_loader.num_val_scenes)]
            for b in range(int(data_loader.num_val_scenes/args.batch_size)):

                xo, xp, xoo, xpo, did = data_loader.next_batch_valid(traj_list[b * args.batch_size:(b + 1) * args.batch_size])
                mo = make_map_batch(xo, did, data_loader.map, args.map_size)

                # # train rnn
                feed = {model.gt_traj_enc: xoo,
                        model.gt_traj_dec: xpo,
                        model.gt_map_enc: mo,
                        model.init_state_enc: state,
                        model.output_keep_prob: 1.0}

                # run prediction
                valid_loss = sess.run(model.cost_valid, feed)
                valid_loss_list.append(valid_loss)

            # show current performance
            print('>> cur cost: %.4f, cur best: %.4f (time %.1fs, p-lvl %02d, %.1f hrs left)'
                  % (np.mean(valid_loss_list), min_avg_loss, (end - start), patient, ((end - start) * (args.num_epochs - e - 1) / 3600.0)))

            # save every breakthrough
            if (min_avg_loss > np.mean(valid_loss_list)):
                best_epoch = e
                patient = 0
                min_avg_loss = np.mean(valid_loss_list)
                checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print(">> model saved to {}".format(checkpoint_path))

                with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
                    args.start_epoch = e
                    args.min_avg_loss = min_avg_loss
                    pickle.dump(args, f)
                print(">> args.start_epoch & args.min_avg_loss are updated and saved to {}".format('config.pkl'))
            else:
                patient += 1

            # # Save log
            if e % 10 == 0:
                file_name_txt = 'current_progress_' + str(e) + '.txt'
                file = open(os.path.join(args.model_dir, file_name_txt), "w")
                file.write('current best val. loss: ' + str(min_avg_loss) + '\n')
                file.write('p-evel: ' + str(patient) + '\n')
                file.write('remaining time: ' + str(((end - start) * (args.num_epochs - e - 1) / 3600.0)) + '\n')
                file.close()

            # early stop
            if (patient > args.patient_thr):
                print_training_info(args, min_avg_loss)
                print('>> early stop triggered ...')
                print('>> Best performance occured at %d epoch, corresponding avg. loss %.4f' % (
                best_epoch, min_avg_loss))
                print('>> goodbye ...')
                break


if __name__ == '__main__':
    main()
