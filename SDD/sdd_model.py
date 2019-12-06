from sdd_functions import *

class Model():

    def __init__(self, args, infer=False):


        # # ------------------------------------------------------------------
        # # Parameter setting
        print('>> network configuration starts ...')

        # define training or validation mode
        if infer:
            args.batch_size = 1
        self.args = args

        # in pose & out pose dim.
        self.input_dim = args.input_dim
        self.output_dim = args.input_dim
        self.pred_length = args.pred_length

        # semantic map info
        self.map_size = args.map_size

        # convnet info
        self.conv_flat_size = 900

        # reward function in & out info
        self.fc_size_in = self.conv_flat_size + self.input_dim
        self.fc_size_out = 1


        # # ------------------------------------------------------------------
        # # Define network structure

        def get_cell(name):
            return tf.contrib.rnn.BasicLSTMCell(args.rnn_size, state_is_tuple=False, name=name)

        cell_enc = get_cell('encoder')
        cell_dec = get_cell('decoder')

        # add dropout layer when training
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name='output_keep_prob')

        cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=self.output_keep_prob)
        cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=self.output_keep_prob)
        self.cell_enc = cell_enc
        self.cell_dec = cell_dec

        # define tensor for cell states
        zero_state_enc = tf.split(tf.zeros([args.batch_size, cell_enc.state_size]), axis=0, num_or_size_splits=args.batch_size)
        zero_state_dec = tf.split(tf.zeros([args.batch_size, cell_dec.state_size]), axis=0, num_or_size_splits=args.batch_size)
        self.init_state_enc = tf.identity(zero_state_enc, name='init_state_enc')

        # output states : batch_size x (1 x args.rnn_size)
        self.output_states_enc = tf.split(tf.zeros([args.batch_size, args.rnn_size]), axis=0, num_or_size_splits=args.batch_size)
        self.output_states_dec = tf.split(tf.zeros([args.batch_size, args.rnn_size]), axis=0, num_or_size_splits=args.batch_size)


        # # -----------------------------------------------------------------------------
        # # Define variables for training

        # placeholders for encoder
        self.gt_traj_enc = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.obs_length, self.input_dim], name='gt_traj_enc')
        self.gt_map_enc = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.obs_length, self.map_size, self.map_size, 3], name='gt_map_enc')
        self.policy_map_enc = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.obs_length, self.map_size, self.map_size, 3], name='policy_map_enc')

        # placeholders for decoder
        self.gt_traj_dec = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.pred_length, self.input_dim], name='gt_traj_dec')


        # initialize cost
        # TODO : do we need to introduce discount factor?
        # beta = tf.constant(1.00, name="discount_factor")
        self.cost_reward = tf.constant(0.0, name="cost_reward")
        self.cost_policy = tf.constant(0.0, name="cost_policy")
        self.reward_gt_avg = tf.constant(0.0, name="rgt_avg")
        self.reward_policy_avg = tf.constant(0.0, name="rpol_avg")
        self.cost_pos_dec = tf.constant(0.0, name="cost_pos_dec")
        self.cost_valid = tf.constant(0.0, name="cost_valid")



        with tf.variable_scope('convlayer'):

            # conv layer
            # TODO : what about increase conv layer size or use pretrained VGGNET
            cw1, cb1 = initialize_conv_filter(shape=[5, 5, 3, 6], name='conv1')
            cw2, cb2 = initialize_conv_filter(shape=[3, 3, 6, 9], name='conv2')
            cw3, cb3 = initialize_conv_filter(shape=[3, 3, 9, 9], name='conv3')


        with tf.variable_scope('reward'):

            # TODO : what about increase reward function size
            fwr = weight_variable(shape=[self.fc_size_in, self.fc_size_out], stddev=0.01, name='fw3')
            fbr = bias_variable(shape=[self.fc_size_out], init=0.0, name='fb3_bias')


        with tf.variable_scope('rnnlm'):

            # embedding for pose input (encoder)
            embedding_we = weight_variable(shape=[self.input_dim, int(args.rnn_size/2)], stddev=0.01, name='embedding_we')
            embedding_be = bias_variable(shape=[int(args.rnn_size/2)], init=0.0, name='embedding_be_bias')

            # embedding for conv output (encoder)
            embedding_wc = weight_variable(shape=[self.conv_flat_size, int(args.rnn_size/2)], stddev=0.01, name='embedding_wc')
            embedding_bc = bias_variable(shape=[int(args.rnn_size/2)], init=0.0, name='embedding_bc_bias')

            # embedding for encoder concatenate
            embedding_wcc = weight_variable(shape=[args.rnn_size, args.rnn_size], stddev=0.01, name='embedding_wcc')
            embedding_bcc = bias_variable(shape=[args.rnn_size], init=0.0, name='embedding_bcc_bias')

            # embedding for pose input (decoder)
            embedding_wd = weight_variable(shape=[self.input_dim, int(args.rnn_size)], stddev=0.01, name='embedding_wd')
            embedding_bd = bias_variable(shape=[int(args.rnn_size)], init=0.0, name='embedding_bd_bias')

            # output for encoder
            output_we = weight_variable(shape=[args.rnn_size, self.output_dim], stddev=0.01, name='output_we')
            output_be = bias_variable(shape=[self.output_dim], init=0.0, name='output_be_bias')

            # output for decoder
            output_wd = weight_variable(shape=[args.rnn_size, self.output_dim], stddev=0.01, name='output_wd')
            output_bd = bias_variable(shape=[self.output_dim], init=0.0, name='output_bd_bias')


        # # ----------------------------------------------------------------------
        # Processing map info

        # gt map for encoder
        conv_out_gt_enc = tf.unstack(tf.zeros(shape=[args.batch_size, args.obs_length, self.conv_flat_size]), axis=1)  # obs_length x (batch_size x fc_size_out)
        map_batches_gt_enc = tf.unstack(self.gt_map_enc, axis=1) # obs_length x (batch_size x map_size x map_size x 3)

        conv_out_policy_enc = tf.unstack(tf.zeros(shape=[args.batch_size, args.obs_length, self.conv_flat_size]), axis=1)
        mpa_batches_policy_enc = tf.unstack(self.policy_map_enc, axis=1)

        for sidx in range(args.obs_length):
            map_batch_gt_enc = map_batches_gt_enc[sidx]  # (batch_size x map_size x map_size x 3)
            conv_out_gt_enc[sidx] = shallow_convnet(map_batch_gt_enc, cw1, cb1, cw2, cb2, cw3, cb3)  # (batch_size x fc_size_out)

            map_batch_policy_enc = mpa_batches_policy_enc[sidx]  # (batch_size x map_size x map_size x 3)
            conv_out_policy_enc[sidx] = shallow_convnet(map_batch_policy_enc, cw1, cb1, cw2, cb2, cw3, cb3)  # (batch_size x fc_size_out)

        # reshape convout sequences
        conv_seqs_gt_enc = tf.unstack(tf.stack(conv_out_gt_enc, axis=1), axis=0)          # batch_size x (obs_length x fc_size_out)
        conv_seqs_policy_enc = tf.unstack(tf.stack(conv_out_policy_enc, axis=1), axis=0)  # batch_size x (obs_length x fc_size_out)


        # # ----------------------------------------------------------------------
        # Processing pose info

        # batch_size x (seq_length x input_dim)
        input_seqs_enc = tf.unstack(self.gt_traj_enc, axis=0)
        input_seqs_dec = tf.unstack(self.gt_traj_dec, axis=0)

        # batch_size x (seq_length x embedding_size)
        embedding_seqs_enc = tf.unstack(tf.zeros(shape=[args.batch_size, args.obs_length, args.rnn_size]), axis=0)

        # embedding operation
        for i in range(args.batch_size):

            # pose embedding
            embedding_pose_enc = tf.nn.relu(tf.nn.xw_plus_b(input_seqs_enc[i], embedding_we, embedding_be))

            # map embedding
            embedding_conv_enc = tf.nn.relu(tf.nn.xw_plus_b(conv_seqs_gt_enc[i], embedding_wc, embedding_bc))

            # concatenate
            embedding_concat_enc = tf.concat([embedding_pose_enc, embedding_conv_enc], axis=1)

            # embedding
            embedding_seqs_enc[i] = tf.nn.relu(tf.nn.xw_plus_b(embedding_concat_enc, embedding_wcc, embedding_bcc))


        # # ----------------------------------------------------------------------
        # data goes through RNN Encoder
        self.predictions_enc = []
        for b in range(args.batch_size):

            # current embedding seqs
            cur_embed_seq_enc = embedding_seqs_enc[b]

            # ------------ reward related -------------#
            # current gt pose
            cur_gt_pose_seq_enc = input_seqs_enc[b]

            # ground-truth map
            cur_gt_convout_seq_enc = conv_seqs_gt_enc[b]

            # map from plicy
            cur_policy_convout_seq_dec = conv_seqs_policy_enc[b]
            # ------------ reward related -------------#

            self.reward_gt_avg = 0.0
            self.reward_policy_avg = 0.0
            prev_pred_pose_enc = tf.zeros(shape=[1, args.input_dim])
            for f in range(args.obs_length):

                # current embedding frame
                cur_embed_frm_enc = tf.reshape(cur_embed_seq_enc[f], shape=(1, args.rnn_size))

                # ------------ reward related -------------#
                # cur map frame
                cur_gt_convout_frm_enc = tf.reshape(cur_gt_convout_seq_enc[f], shape=(1, self.conv_flat_size))
                cur_policy_convout_frm_enc = tf.reshape(cur_policy_convout_seq_dec[f], shape=(1, self.conv_flat_size))

                # cur gt pose
                cur_gt_pose_enc = tf.reshape(cur_gt_pose_seq_enc[f], shape=(1, args.input_dim))
                # ------------ reward related -------------#

                with tf.variable_scope("rnnlm") as scope:
                    if (b>0 or f>0):
                        scope.reuse_variables()

                    # go through RNN encoder
                    self.output_states_enc[b], zero_state_enc[b] = cell_enc(cur_embed_frm_enc, zero_state_enc[b])

                    # fully connected layer for output
                    cur_pred_pose_enc = tf.nn.xw_plus_b(self.output_states_enc[b], output_we, output_be)
                    self.predictions_enc.append(cur_pred_pose_enc)

                    # ------------ reward related -------------#
                    if (f > 1):
                        self.reward_gt_avg += calculate_reward(fwr, fbr, cur_gt_convout_frm_enc, cur_gt_pose_enc)
                        self.reward_policy_avg += calculate_reward(fwr, fbr, cur_policy_convout_frm_enc, prev_pred_pose_enc)
                    # ------------ reward related -------------#

                    # delay
                    prev_pred_pose_enc = cur_pred_pose_enc

            # ------------ reward related -------------#
            self.reward_gt_avg /= (args.obs_length-2)
            self.reward_policy_avg /= (args.obs_length-2)

            self.cost_reward += -1.0 * tf.log(self.reward_gt_avg - self.reward_policy_avg + 1.0 + 1e-20)
            self.cost_policy += tf.log(self.reward_gt_avg - self.reward_policy_avg + 1.0 + 1e-20)
            # ------------ reward related -------------#


        # # --------------------------------------------------------------------
        # prediction by RNN decoder
        self.predictions_dec = []

        # For each sequence in the input batch
        for b in range(args.batch_size):

            ''' for the initial input of decoder RNN, use the last input of encoder RNN'''
            cur_input_seq_enc = input_seqs_enc[b]
            init_pose_dec = tf.reshape(cur_input_seq_enc[args.obs_length - 1], shape=(1, args.input_dim)) # xi

            # ground-truth traj
            cur_gt_traj_dec = input_seqs_dec[b]

            # For each frame in a sequence
            for f in range(args.pred_length):

                with tf.variable_scope("rnnlm") as scope:
                    if (f == 0):
                        cur_embed_frm_dec = tf.nn.relu(tf.nn.xw_plus_b(init_pose_dec, embedding_wd, embedding_bd))
                        zero_state_dec[b] = tf.reshape(zero_state_enc[b], shape=(1, 2*args.rnn_size))
                    else:
                        scope.reuse_variables()

                    # go through RNN decoder
                    self.output_states_dec[b], zero_state_dec[b] = cell_dec(cur_embed_frm_dec, zero_state_dec[b])

                    # fully connected layer for output
                    cur_pred_pose_dec = tf.nn.xw_plus_b(self.output_states_dec[b], output_wd, output_bd)
                    self.predictions_dec.append(cur_pred_pose_dec)

                    # corresponding ground truth
                    cur_gt_pose_dec = tf.reshape(cur_gt_traj_dec[f], shape=(1, args.input_dim))

                    # go through embedding function for the next input
                    cur_embed_frm_dec = tf.nn.relu(tf.nn.xw_plus_b(cur_pred_pose_dec, embedding_wd, embedding_bd))

                    # calculate MSE loss
                    mse_loss = tf.reduce_sum(tf.pow(tf.subtract(cur_pred_pose_dec, cur_gt_pose_dec), 2.0))

                    self.cost_pos_dec += mse_loss
                    self.cost_valid += mse_loss

        print('>> network configuration is done ...')


        # # --------------------------------------------------------------------
        # Define final cost and optimizer

        if (infer == False):

            # normalize cost
            self.cost_pos_dec /= (args.batch_size * args.pred_length)
            self.cost_valid /= (args.batch_size * args.pred_length)
            self.cost_reward /= args.batch_size
            self.cost_policy /= args.batch_size

            # gather all the trainable weights
            tvars = tf.trainable_variables()

            # trainable variables in conv layer
            tvars_conv = [var for var in tvars if 'convlayer' in var.name]

            # trainable variables in reward layer
            tvars_reward = [var for var in tvars if 'reward' in var.name]

            # trainable variables in rnn layer
            tvars_pose = [var for var in tvars if 'rnnlm' in var.name]


            # for l2-regularization (bias term is excluded)
            l2_conv = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_conv if not ("bias" in tvar.name))
            l2_reward = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_reward if not ("bias" in tvar.name))
            l2_pose = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_pose if not ("bias" in tvar.name))


            # debug : verification code ----
            #print('conv l2')
            #[print(tvar) for tvar in tvars_conv if not ("bias" in tvar.name)]
            #print('reward l2')
            #[print(tvar) for tvar in tvars_reward if not ("bias" in tvar.name)]
            #print('pose l2')
            #[print(tvar) for tvar in tvars_pose if not ("bias" in tvar.name)]

            # conv layer needs to be trained while training reward layer and rnn layer
            tvars_conv_reward = copy.copy(tvars_reward)
            tvars_conv_pose = copy.copy(tvars_pose)
            for var in tvars_conv:
                tvars_conv_reward.append(var)
                tvars_conv_pose.append(var)


            # add to overall cost
            # TODO : what about define three costs respctively for pose, encoder reward, reward function
            self.cost_pos_dec += (args.gamma_param)*self.cost_policy + l2_pose + l2_conv
            self.cost_reward += l2_reward + l2_conv

            # gradient clipping
            grads_pose, _ = tf.clip_by_global_norm(tf.gradients(self.cost_pos_dec, tvars_conv_pose), args.grad_clip)
            grads_reward = tf.gradients(self.cost_reward, tvars_conv_reward)

            # TODO : what about define separate optimizers for generator and reward function (Social GAN did it)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)

            # define train operation
            self.train_op_pose = optimizer.apply_gradients(zip(grads_pose, tvars_conv_pose))
            self.train_op_reward = optimizer.apply_gradients(zip(grads_reward, tvars_conv_reward))


    def sample(self, sess, xoo, mo, init_state_enc):

        feed = {self.gt_traj_enc: xoo,
                self.gt_map_enc: mo,
                self.init_state_enc: init_state_enc,
                self.output_keep_prob: 1.0}

        pred_offset = sess.run(self.predictions_dec, feed)
        est_offset = np.array(pred_offset).reshape(self.pred_length, self.args.input_dim)

        return est_offset
