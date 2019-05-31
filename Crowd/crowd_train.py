import argparse
import time

from crowd_model import Model
from crowd_utils import DataLoader
from crowd_functions import *
from crowd_occupancy import *


def print_training_info(args, min_avg_loss):
    print('-----------------------------------------------------------')
    print('------------ Encoder-Decoder LSTM INFORMATION -------------')
    print('.network structure: LSTM')
    print('   rnn size (%d)' % (args.rnn_size))
    print('   num layers (%d)' % (args.num_layers))
    print('.dataset path')
    print('   ' + args.dataset_path)
    print('.training setting')
    print('   num epochs (%d)' % args.num_epochs)
    print('   batch size (%d)' % args.batch_size)
    print('   obs_length (%d)' % args.obs_length)
    print('   pred_length (%d)' % args.pred_length)
    print('   learning rate (%.5f)' % args.learning_rate)
    print('   reg. lambda (%.5f)' % args.lambda_param)
    print('   gamma param (%.5f)' % args.gamma_param)
    print('   grad_clip (%.2f)' % args.grad_clip)
    print('   keep prob (%.2f)' % args.keep_prob)
    print('   max num peds (%d)' % args.max_num_peds)
    print('   data load step (%d)' % args.data_load_step)
    print('   is pretrain (%d)' % args.is_pretrain)
    print('   load pretrained (%d)' % args.load_pretrained)

    if (min_avg_loss < 0):
        print('.minimum average loss for validation : %.4f' % min_avg_loss)

    print('   gpu id used (%d)' % args.gpu_num)
    print('------------------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()

    # # Is print information ?
    parser.add_argument('--isprint', type=int, default=0,
                        help='determine if you want to print specific info.')

    # # network structure : LSTM
    parser.add_argument('--rnn_size', type=int, default=32,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--input_dim', type=int, default=2,
                        help='dimension of input vector')

    # # training setting
    '''
    leave dataset | dataset number
    eth           |       0
    hotel         |       1
    univ          |       2
    zara1         |       3
    zara2         |       4
    '''
    parser.add_argument('--exp_id', type=int, default=0, # ------------------
                        help='experiment id')
    parser.add_argument('--dataset_num', type=int, default=3, # ------------------
                        help='target dataset number')
    parser.add_argument('--dataset_path', type=str, default='./datasets/',
                        help='dataset path')
    parser.add_argument('--is_pretrain', type=int, default=0, # ------------------
                        help='this is pretrain using smoothed dataset?')
    parser.add_argument('--load_pretrained', type=int, default=1, # ------------------
                        help='want to load pre-trained network?')
    parser.add_argument('--batch_size', type=int, default=1,  # ------------------
                        help='minibatch size')
    parser.add_argument('--obs_length', type=int, default=8,
                        help='observation sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction sequence length')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=1.5,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lambda_param', type=float, default=0.0005, # ------------------
                        help='regularization weight')
    parser.add_argument('--gamma_param', type=float, default=0.0001, # ------------------
                        help='regularization weight2')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--patient_thr', type=float, default=100,
                        help='threshold for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='default gpu number')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch value')
    parser.add_argument('--min_avg_loss', type=float, default=100000.0,
                        help='min avg loss')
    parser.add_argument('--data_load_step', type=int, default=2, # ------------------
                        help='data_load_step')


    # # social info
    parser.add_argument('--max_num_peds', type=int, default=2, # ------------------
                        help='maximum number of peds')
    parser.add_argument('--social_range', type=int, default=2,
                        help='maximum distance for considering social neighbor')
    parser.add_argument('--grid_size', type=int, default=8,
                        help='grid size')

    args = parser.parse_args()
    train(args)


def train(args):

    # assign GPU device
    if (args.gpu_num == 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif (args.gpu_num == 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif (args.gpu_num == 2):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    elif (args.gpu_num == 3):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # print training information
    print_training_info(args, 0.0)

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
        for e in range(start_epoch, args.num_epochs):
        #for e in range(2):

            # initialize state
            state = model.init_states_enc.eval()

            # # Train one epoch -------------------------------------------------------
            printProgressBar(0, data_loader.num_batches - 1, prefix='Progress:', suffix='Complete', length=50, epoch=e)

            # make random batch list
            rand_list = [x for x in range(data_loader.num_train_scenes)]
            random.shuffle(rand_list)

            start = time.time()
            for b in range(data_loader.num_batches):
            #for b in range(1):

                '''
                np : num peds       <1> x batch_size
                xo : obs seq        <max_num_peds x obs_len x 2>  x batch_size
                xp : pred seq       <max_num_peds x pred_len x 2>  x batch_size
                xoo : obs offsets   <max_num_peds x obs_len x 2>  x batch_size
                xpo : pred offsets  <max_num_peds x pred_len x 2>  x batch_size
                xm : loss mask      <max_num_peds x 1>  x batch_size
                '''

                # step 1) make mini-batch for training (checked)

                # load mini-batch
                nps, xo, xp, xoo, xpo, xm = data_loader.next_batch(rand_list[b*args.batch_size:(b+1)*args.batch_size])

                # create occupancy grid map (ogm)
                ogm = rectangular_occupancy_map(xo, args.max_num_peds, nps, args.obs_length, args.social_range, args.grid_size)

                # conversion to np array
                nps = np.array(nps).reshape(args.batch_size, 1)
                xoo = np.array(xoo).reshape(args.batch_size, args.max_num_peds, args.obs_length, args.input_dim)
                xpo = np.array(xpo).reshape(args.batch_size, args.max_num_peds, args.pred_length, args.input_dim)
                xm = np.array(xm).reshape(args.batch_size, args.max_num_peds, 1)

                # step 2) draw samples from encoder and update occupancy grid map (checked)
                '''
                xoo = [x0, x1, x2, x3, x4]
                xoo_p = estimate of [x1, x2, x3, x4, x5]
                '''

                # past traj prediction by RNN encoder
                feed = {model.gt_trajs_enc: xoo,
                        model.gt_ogm_enc: ogm,
                        model.num_valid_peds: nps,
                        model.init_states_enc: state,
                        model.output_keep_prob: 1.0}
                xoo_p = sess.run([model.predictions_enc], feed)

                # reshaping
                xoo_p_re = np.squeeze(np.array(xoo_p)).reshape(args.obs_length, args.batch_size, args.max_num_peds, args.input_dim)
                xoo_p_re = np.swapaxes(np.swapaxes(xoo_p_re, 0, 1), 1, 2) # batch x mnp x obs_len x input

                # create occupancy grid map (ogm) for policy
                # debug 190110 ----
                ogm_p = rectangular_occupancy_map_policy(xo, xoo, xoo_p_re, args.max_num_peds, nps, args.obs_length, args.social_range, args.grid_size)


                # step 3) train policy (RNN)
                feed = {model.gt_trajs_enc: xoo,
                        model.gt_trajs_dec: xpo,
                        model.loss_mask: xm,
                        model.gt_ogm_enc: ogm,
                        model.pol_ogm_enc: ogm_p,
                        model.num_valid_peds: nps,
                        model.init_states_enc: state,
                        model.output_keep_prob: 0.8}
                train_loss, _ = sess.run([model.cost_pos_dec, model.train_op_pose], feed)


                # step 4) train reward function
                train_loss, _ = sess.run([model.cost_reward, model.train_op_reward], feed)

                printProgressBar(b, data_loader.num_batches - 1, prefix='Progress:', suffix='Complete', length=50, epoch=e)

            end = time.time()


            # # Validation ------------------------------------------------------------
            valid_loss_list = []

            # initialize state
            state = model.init_states_enc.eval()
            traj_list = [x for x in range(data_loader.num_val_scenes)]
            for b in range(int(data_loader.num_val_scenes/args.batch_size)):

                nps, xo, xp, xoo, xpo, xm = data_loader.next_batch_valid(traj_list[b * args.batch_size:(b + 1) * args.batch_size])

                ogm = rectangular_occupancy_map(xo, args.max_num_peds, nps, args.obs_length, args.social_range, args.grid_size)

                # conversion to array
                nps = np.array(nps).reshape(args.batch_size, 1)
                xoo = np.array(xoo).reshape(args.batch_size, args.max_num_peds, args.obs_length, args.input_dim)
                xpo = np.array(xpo).reshape(args.batch_size, args.max_num_peds, args.pred_length, args.input_dim)
                xm = np.array(xm).reshape(args.batch_size, args.max_num_peds, 1)

                feed = {model.gt_trajs_enc: xoo,
                        model.gt_trajs_dec: xpo,
                        model.loss_mask: xm,
                        model.gt_ogm_enc: ogm,
                        model.num_valid_peds: nps,
                        model.init_states_enc: state,
                        model.output_keep_prob: 1.0}

                # run prediction
                valid_loss = sess.run(model.cost_valid, feed)
                valid_loss_list.append(valid_loss)

            # show current performance
            print('>> cost: %.4f, cur. best: %.4f (time %.1fs, p-lvl %02d, %.1f hrs left)'
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
                file.write('remaining time: ' + str(((end-start)*(args.num_epochs-e-1)/3600.0)) + '\n')
                file.close()

            # early stop
            if (patient > args.patient_thr):
                print_training_info(args, min_avg_loss)
                print('>> early stop triggered ...')
                print('>> Best performance occured at %d epoch, corresponding avg. MAE %.4f' % (best_epoch, min_avg_loss))
                print('>> goodbye ...')
                break


if __name__ == '__main__':
    main()
