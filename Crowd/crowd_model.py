from crowd_functions import *

class Model():

    def __init__(self, args, infer=False):


        # # ------------------------------------------------------------------
        # # Parameter setting
        print('>> network configuration starts ...')
        if infer:
            args.batch_size = 1

        # define training or validation mode
        self.args = args

        # in pose & out pose dim.
        self.input_dim = args.input_dim
        self.output_dim = args.input_dim

        self.rnn_size = args.rnn_size
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.MNP = args.max_num_peds
        self.batch_size = args.batch_size
        self.grid_size = args.grid_size

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


        # # -----------------------------------------------------------------------------
        # # Placeholders
        '''
        xoo : batch_size x max_num_peds x obs_len x input_dim
        xpo : batch_size x max_num_peds x pred_len x input_dim
        nps : batch_size x 1
        xm : batch_size x max_num_peds x 1
        '''

        # hidden states : batch_size x <MNP x 2*rnn_size>
        zero_states_enc = tf.unstack(tf.zeros(shape=[args.batch_size, self.MNP, cell_enc.state_size]), axis=0)
        zero_states_dec = tf.unstack(tf.zeros(shape=[args.batch_size, self.MNP, cell_enc.state_size]), axis=0)
        self.init_states_enc = tf.identity(zero_states_enc, name='init_state_enc')

        # output states : batch_size x <MNP x rnn_size>
        output_states_enc = tf.unstack(tf.zeros(shape=[args.batch_size, self.MNP, args.rnn_size]), axis=0)
        output_states_dec = tf.unstack(tf.zeros(shape=[args.batch_size, self.MNP, args.rnn_size]), axis=0)

        # holders for prev dec output embeddings : : batch_size x <MNP x rnn_size>
        prev_embed_frms_dec = tf.unstack(tf.zeros([args.batch_size, self.MNP, args.rnn_size]), axis=0)

        # place holders
        self.gt_trajs_enc= tf.placeholder(dtype=tf.float32, shape=[args.batch_size, self.MNP, args.obs_length, self.input_dim], name='gt_traj_enc')
        self.gt_trajs_dec = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, self.MNP, args.pred_length, self.input_dim], name='gt_traj_dec')

        self.gt_ogm_enc = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.obs_length, self.MNP, self.MNP, self.grid_size**2], name='gt_traj_dec')
        # ------------ reward related -------------#
        self.pol_ogm_enc = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.obs_length, self.MNP, self.MNP, self.grid_size ** 2], name='pol_traj_dec')
        # ------------ reward related -------------#

        self.loss_mask = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, self.MNP, 1], name='loss_mask')
        self.num_valid_peds = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 1], name='num_valid_peds')

        # note : re-shaping states to <batch_size x MNP x something>
        zero_states_enc_list = []
        zero_states_dec_list = []
        output_states_enc_list = []
        output_states_dec_list = []
        prev_embed_frms_dec_list = []

        # ------------ reward related -------------#
        prev_pred_pose_enc_list = []
        # ------------ reward related -------------#

        for b in range(args.batch_size):
            ped_state_list_enc = []
            ped_state_list_dec = []
            ped_output_list_enc = []
            ped_output_list_dec = []
            ped_embed_list_dec = []

            # ------------ reward related -------------#
            ped_pose_list_enc = []
            # ------------ reward related -------------#

            # max_num_peds x (2*rnn_size) or max_num_peds x (rnn_size)
            zero_state_enc = tf.unstack(zero_states_enc[b], axis=0)
            zero_state_dec = tf.unstack(zero_states_dec[b], axis=0)
            output_state_enc = tf.unstack(output_states_enc[b], axis=0)
            output_state_dec = tf.unstack(output_states_dec[b], axis=0)
            ped_embed_dec = tf.unstack(prev_embed_frms_dec[b], axis=0)

            for p in range(self.MNP):
                ped_state_list_enc.append(tf.reshape(zero_state_enc[p], shape=(1, 2*args.rnn_size))) # 1 x 2*rnn_size
                ped_state_list_dec.append(tf.reshape(zero_state_dec[p], shape=(1, 2 * args.rnn_size)))  # 1 x 2*rnn_size
                ped_output_list_enc.append(tf.reshape(output_state_enc[p], shape=(1, args.rnn_size)))
                ped_output_list_dec.append(tf.reshape(output_state_dec[p], shape=(1, args.rnn_size)))
                ped_embed_list_dec.append(tf.reshape(ped_embed_dec[p], shape=(1, args.rnn_size)))

                # ------------ reward related -------------#
                ped_pose_list_enc.append(tf.zeros(shape=[1, args.input_dim]))
                # ------------ reward related -------------#
                
            zero_states_enc_list.append(ped_state_list_enc)
            output_states_enc_list.append(ped_output_list_enc)
            zero_states_dec_list.append(ped_state_list_dec)
            output_states_dec_list.append(ped_output_list_dec)
            prev_embed_frms_dec_list.append(ped_embed_list_dec)

            # ------------ reward related -------------#
            prev_pred_pose_enc_list.append(ped_pose_list_enc)
            # ------------ reward related -------------#


        # initialize cost
        self.cost_pos_dec = tf.constant(0.0, name="cost_pos_dec")
        self.cost_valid = tf.constant(0.0, name="cost_valid")

        # ------------ reward related -------------#
        self.cost_reward = tf.constant(0.0, name="cost_reward")
        self.cost_policy = tf.constant(0.0, name="cost_policy")
        self.reward_gt_avg = tf.constant(0.0, name="rgt_avg")
        self.reward_policy_avg = tf.constant(0.0, name="rpol_avg")
        # ------------ reward related -------------#


        # # -----------------------------------------------------------------------------
        # # Define variables for training


        with tf.variable_scope('analyzer'):

            # embedding for pooled tensor (encoder)
            embedding_wes = weight_variable(shape=[args.rnn_size*self.grid_size*self.grid_size, args.rnn_size], stddev=0.01, name='embedding_wes')
            embedding_bes = bias_variable(shape=[args.rnn_size], init=0.0, name='embedding_bes_bias')

        with tf.variable_scope('reward'):

            # reward function
            embedding_wr = weight_variable(shape=[args.rnn_size+args.input_dim, 1], stddev=0.01, name='embedding_wr')
            embedding_br = bias_variable(shape=[1], init=0.0, name='embedding_br_bias')

        with tf.variable_scope('rnnlm'):

            # embedding for pose input (encoder)
            embedding_we = weight_variable(shape=[self.input_dim, args.rnn_size], stddev=0.01, name='embedding_we')
            embedding_be = bias_variable(shape=[args.rnn_size], init=0.0, name='embedding_be_bias')

            # embedding for mixing 'pooled tensor' and 'input embedding(encoder)'
            embedding_wem = weight_variable(shape=[2*args.rnn_size, args.rnn_size], stddev=0.01, name='embedding_wem')
            embedding_bem = bias_variable(shape=[args.rnn_size], init=0.0, name='embedding_bem_bias')

            # embedding for pose input (decoder)
            embedding_wd = weight_variable(shape=[self.input_dim, args.rnn_size], stddev=0.01, name='embedding_wd')
            embedding_bd = bias_variable(shape=[args.rnn_size], init=0.0, name='embedding_bd_bias')

            # output for encoder
            output_we = weight_variable(shape=[args.rnn_size, self.output_dim], stddev=0.01, name='output_we')
            output_be = bias_variable(shape=[self.output_dim], init=0.0, name='output_be_bias')

            # output for decoder
            output_wd = weight_variable(shape=[args.rnn_size, self.output_dim], stddev=0.01, name='output_wd')
            output_bd = bias_variable(shape=[self.output_dim], init=0.0, name='output_bd_bias')


        # # ----------------------------------------------------------------------
        # Processing pose info

        # batch_size x (max_num_peds x seq_length x input_dim)
        input_seqs_enc = tf.unstack(self.gt_trajs_enc, axis=0)
        input_seqs_dec = tf.unstack(self.gt_trajs_dec, axis=0)

        # max_num_peds x (seq_length x embedding_size)
        embedding_peds_seq_enc = tf.unstack(tf.zeros(shape=[self.MNP, args.obs_length, args.rnn_size]), axis=0)

        # batch_size x (max_num_peds x seq_length x embedding_size)
        embedding_seqs_enc = tf.unstack(tf.zeros(shape=[args.batch_size, self.MNP, args.obs_length, args.rnn_size]), axis=0)

        # embedding operation
        for i in range(args.batch_size):

            # max_num_peds x (seq_length x input_dim)
            cur_seq_enc = tf.unstack(input_seqs_enc[i], axis=0)

            for j in range(self.MNP):

                # seq_length x input_dim
                cur_ped_seq_enc = cur_seq_enc[j]

                # embedding
                embedding_peds_seq_enc[j] = tf.nn.relu(tf.nn.xw_plus_b(cur_ped_seq_enc, embedding_we, embedding_be))

            # insert
            embedding_seqs_enc[i] = tf.stack(embedding_peds_seq_enc, axis=0)


        # # ----------------------------------------------------------------------
        # step 1) data goes through RNN Encoder

        # seq_length x (batch_size x max_num_peds  x embedding_size)
        embedding_seqs_enc = tf.unstack(tf.stack(embedding_seqs_enc, axis=0), axis=2)

        # obs_len x <batch_size x MNP x MNP x grid_size**2>
        ogms_enc = tf.unstack(self.gt_ogm_enc, axis=1)

        # ------------ reward related -------------#
        # obs_len x <batch_size x MNP x MNP x grid_size**2>
        ogms_pol_enc = tf.unstack(self.pol_ogm_enc, axis=1)

        # positions {seq_length x (batch_size x max_num_peds  x input_dim)}
        gt_trajs = tf.unstack(self.gt_trajs_enc, axis=2)

        # batch_size x (max_num_peds x 1)
        loss_mask_batch = tf.unstack(self.loss_mask, axis=0)
        # ------------ reward related -------------#

        # FOR EACH FRAME INDEX
        self.predictions_enc = []
        for f in range(args.obs_length):

            # batch_size x (max_num_peds x embedding_size) okay
            cur_embed_frms_enc = tf.unstack(embedding_seqs_enc[f], axis=0)

            # batch_size x <MNP x MNP x grid_size**2>
            ogm_peds_enc = tf.unstack(ogms_enc[f], axis=0)

            # ------------ reward related -------------#
            # batch_size x <MNP x MNP x grid_size**2>
            ogm_pol_peds_enc = tf.unstack(ogms_pol_enc[f], axis=0)

            # batch_size x <MNP x input_dim>
            cur_gt_trajs = tf.unstack(gt_trajs[f], axis=0)
            # ------------ reward related -------------#

            # FOR EACH ELEMENT IN A BATCH
            for b in range(args.batch_size):

                # ------------ reward related -------------#
                # MNP x (input_dim)
                cur_gt_postions = tf.unstack(cur_gt_trajs[b], axis=0)
                # ------------ reward related -------------#

                # max_num_peds x (embedding_size)
                cur_embed_peds_enc = tf.unstack(cur_embed_frms_enc[b], axis=0)

                # MNP x (1 x rnn_size) okay
                social_tensor = self.GetSocialPooledHiddenStates(ogm_peds_enc[b], output_states_enc_list[b], embedding_wes, embedding_bes)

                # ------------ reward related -------------#
                # MNP x (1 x rnn_size) okay
                social_tensor_pol = self.GetSocialPooledHiddenStates(ogm_pol_peds_enc[b], output_states_enc_list[b], embedding_wes, embedding_bes)

                # max_num_peds x (1)
                loss_masks_enc = tf.unstack(loss_mask_batch[b], axis=0)
                # ------------ reward related -------------#

                with tf.variable_scope("rnnlm") as scope:
                    if (b > 0 or f > 0):
                        scope.reuse_variables()

                    # FOR EACH PED IN THE ELEMENT
                    for p in range(self.MNP):

                        # ------------ reward related -------------#
                        # 1 x input_dim
                        cur_gt_ped_pos = tf.reshape(cur_gt_postions[p], shape=(1, args.input_dim))
                        # ------------ reward related -------------#

                        cur_embed_ped_enc = tf.reshape(cur_embed_peds_enc[p], shape=(1, args.rnn_size))
                        cur_concate = tf.concat([cur_embed_ped_enc, social_tensor[p]], axis=1)
                        complete_embed = tf.nn.relu(tf.nn.xw_plus_b(cur_concate, embedding_wem, embedding_bem))

                        # go through RNN encoder
                        output_states_enc_list[b][p], zero_states_enc_list[b][p] = cell_enc(complete_embed, zero_states_enc_list[b][p])

                        # ------------ reward related -------------#
                        # fully connected layer for encoder output
                        cur_pred_pose_enc = tf.nn.xw_plus_b(output_states_enc_list[b][p], output_we, output_be)
                        self.predictions_enc.append(cur_pred_pose_enc)

                        # loss mask
                        loss_mask_ped_enc = loss_masks_enc[p]

                        if (f > 1):
                            reward_gt = calculate_reward(embedding_wr, embedding_br, social_tensor[p], cur_gt_ped_pos)
                            reward_policy = calculate_reward(embedding_wr, embedding_br, social_tensor_pol[p], prev_pred_pose_enc_list[b][p])

                            self.reward_gt_avg += tf.multiply(reward_gt, loss_mask_ped_enc)
                            self.reward_policy_avg += tf.multiply(reward_policy, loss_mask_ped_enc)

                        # delay
                        prev_pred_pose_enc_list[b][p] = cur_pred_pose_enc
                        # ------------ reward related -------------#

        # ------------ reward related -------------#
        self.reward_gt_avg /= ((args.obs_length-2) * tf.reduce_sum(self.num_valid_peds))
        self.reward_policy_avg /= ((args.obs_length-2) * tf.reduce_sum(self.num_valid_peds))

        self.cost_reward += -1.0 * tf.log(self.reward_gt_avg - self.reward_policy_avg + 1.0 + 1e-20)
        self.cost_policy += tf.log(self.reward_gt_avg - self.reward_policy_avg + 1.0 + 1e-20)
        # ------------ reward related -------------#



        # # --------------------------------------------------------------------
        # step 2) prediction by RNN decoder
        self.predictions_dec = []

        # note : change from batch first to sequence first
        # seq_length x (batch_size x max_num_peds  x input_dim)
        input_seqs_enc = tf.unstack(tf.stack(input_seqs_enc, axis=0), axis=2)
        input_seqs_dec = tf.unstack(tf.stack(input_seqs_dec, axis=0), axis=2)

        # batch_size x (max_num_peds x input_dim)
        init_frms_dec = tf.unstack(input_seqs_enc[args.obs_length-1], axis=0)

        # batch_size x (max_num_peds x 1)
        # loss_mask_batch = tf.unstack(self.loss_mask, axis=0)

        # For each frame in a sequence
        for f in range(args.pred_length):

            # batch_size x (max_num_peds x input_dim)
            cur_gt_frms_dec = tf.unstack(input_seqs_dec[f], axis=0)

            # For each element in a batch
            for b in range(args.batch_size):

                # ground-truth peds
                cur_gt_ped_frm_dec = tf.unstack(cur_gt_frms_dec[b], axis=0) # max_num_peds x (input_dim)

                # loss mask
                loss_masks = tf.unstack(loss_mask_batch[b], axis=0) # max_num_peds x (1)

                # init pos
                init_frm_dec = tf.unstack(init_frms_dec[b], axis=0) # max_num_peds x (1)

                with tf.variable_scope("rnnlm") as scope:
                    if (f>0 or b>0):
                        scope.reuse_variables()

                    # for each ped in an element
                    for p in range(self.MNP):

                        # ground-truth position
                        cur_gt_pose_dec = tf.reshape(cur_gt_ped_frm_dec[p], shape=(1, self.input_dim))

                        # loss mask
                        loss_mask_ped = loss_masks[p]


                        if (f == 0):
                            # note : the last element of encoder input is used for the init. input of decoder
                            init_pose_dec = tf.reshape(init_frm_dec[p], shape=(1, self.input_dim))
                            cur_embed_frm_dec = tf.nn.relu(tf.nn.xw_plus_b(init_pose_dec, embedding_wd, embedding_bd))
                            zero_states_dec_list[b][p] = tf.reshape(zero_states_enc_list[b][p], shape=(1, 2*args.rnn_size))
                        else:
                            cur_embed_frm_dec = prev_embed_frms_dec_list[b][p]

                        # go through RNN decoder
                        output_states_dec_list[b][p], zero_states_dec_list[b][p] = cell_dec(cur_embed_frm_dec, zero_states_dec_list[b][p])

                        # fully connected layer for output
                        cur_pred_pose_dec = tf.nn.xw_plus_b(output_states_dec_list[b][p], output_wd, output_bd)
                        self.predictions_dec.append(cur_pred_pose_dec)

                        # go through embedding function for the next input
                        prev_embed_frms_dec_list[b][p] = tf.reshape(tf.nn.relu(tf.nn.xw_plus_b(cur_pred_pose_dec, embedding_wd, embedding_bd)), shape=(1, args.rnn_size))

                        # calculate MSE loss
                        mse_loss = tf.reduce_sum(tf.pow(tf.subtract(cur_pred_pose_dec, cur_gt_pose_dec), 2.0))
                        self.cost_pos_dec += tf.multiply(mse_loss, loss_mask_ped)
                        self.cost_valid += tf.multiply(mse_loss, loss_mask_ped)


        print('>> network configuration is done ...')


        # # --------------------------------------------------------------------
        # Define final cost and optimizer

        if (infer == False):

            # normalize cost
            self.cost_pos_dec /= (args.pred_length * tf.reduce_sum(self.num_valid_peds))
            self.cost_valid /= (args.pred_length * tf.reduce_sum(self.num_valid_peds))

            # gather all the trainable weights
            tvars = tf.trainable_variables()

            # trainable variables in analyzer layer
            tvars_analyze = [var for var in tvars if 'analyzer' in var.name]

            # trainable variables in reward layer
            tvars_reward = [var for var in tvars if 'reward' in var.name]

            # trainable variables in rnn layer
            tvars_pose = [var for var in tvars if 'rnnlm' in var.name]


            # for l2-regularization (bias term is excluded)
            l2_analyze = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_analyze if not ("bias" in tvar.name))
            l2_reward = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_reward if not ("bias" in tvar.name))
            l2_pose = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_pose if not ("bias" in tvar.name))

            # debug : verification code ----
            #print('conv l2')
            #[print(tvar) for tvar in tvars_analyze if not ("bias" in tvar.name)]
            #print('reward l2')
            #[print(tvar) for tvar in tvars_reward if not ("bias" in tvar.name)]
            #print('pose l2')
            #[print(tvar) for tvar in tvars_pose if not ("bias" in tvar.name)]

            # analyzer needs to be trained while training reward layer and rnn layer
            tvars_anal_reward = copy.copy(tvars_reward)
            tvars_anal_pose = copy.copy(tvars_pose)
            for var in tvars_analyze:
                tvars_anal_reward.append(var)
                tvars_anal_pose.append(var)

            if(args.isprint == 1):
                print('############ Trainable variables : reward ############')
                for var in tvars_anal_reward:
                    print(var)

                print('############ Trainable variables : pose ############')
                for var in tvars_anal_pose:
                    print(var)


            # add to overall cost
            self.cost_pos_dec += (args.gamma_param * self.cost_policy) + l2_pose + l2_analyze
            self.cost_reward += l2_reward + l2_analyze

            # gradient clipping
            grads_pose, _ = tf.clip_by_global_norm(tf.gradients(self.cost_pos_dec, tvars_anal_pose), args.grad_clip)
            grads_reward = tf.gradients(self.cost_reward, tvars_anal_reward)

            optimizer = tf.train.AdamOptimizer(args.learning_rate)

            # define train operation
            self.train_op_pose = optimizer.apply_gradients(zip(grads_pose, tvars_anal_pose))
            self.train_op_reward = optimizer.apply_gradients(zip(grads_reward, tvars_anal_reward))


    def GetSocialPooledHiddenStates(self, grid_map, output_states, w, b):
        '''
        grid_map: MNP x MNP x grid_size**2
        output_states: MNP x <1 x rnn_size>
        w/b : rnn_size*grid_size*grid_size x rnn_size
        :return: MNP x embedding_size
        '''

        # social tensor
        social_tensor = tf.unstack(tf.zeros(shape=[self.MNP, self.rnn_size]), axis=0)

        # rnn_size x MNP
        states_mat = tf.transpose(tf.squeeze(tf.unstack(output_states, axis=0), axis=1), [1, 0])

        # MNP x <MNP x grid_size**2>
        grid_map_ped = tf.unstack(grid_map, axis=0)

        for p in range(self.MNP):

            # 1 x rnn_size*grid_size*grid_size
            pooled_state = tf.reshape(tf.matmul(states_mat, grid_map_ped[p]), shape=[1, self.rnn_size*self.grid_size*self.grid_size])

            # 1 x rnn_size
            social_tensor[p] = tf.nn.relu(tf.nn.xw_plus_b(pooled_state, w, b))

        return social_tensor


    def sample(self, sess, xoo, init_state, ogm):

        # init_state_enc = sess.run(self.cell_enc.zero_state_enc(1, tf.float32))
        # init_state_enc = sess.run(self.init_states_enc)

        '''
        feed = {model.gt_traj_enc: xoo,
                model.gt_traj_dec: xpo,
                model.loss_mask: xm,
                model.num_valid_peds: nps,
                model.init_state_enc: state,
                model.output_keep_prob: 1.0}
        '''
        # ----------------------------------------------------------------
        feed = {self.gt_trajs_enc: xoo,
                self.init_states_enc: init_state,
                self.gt_ogm_enc: ogm,
                self.output_keep_prob: 1.0}

        pred_traj = sess.run(self.predictions_dec, feed)
        est_traj = np.array(pred_traj).reshape(self.pred_length, self.MNP, self.args.input_dim)

        return np.swapaxes(est_traj, 0, 1)
