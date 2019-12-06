'''
 SDD TRACKING DATA LOADER
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''

import pickle
import os
from sdd_functions import *
from sdd_functions import _min, _max


class DataLoader:

    def __init__(self, args):

        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.batch_size_valid = 1
        self.input_dim = args.input_dim
        self.seq_length = args.obs_length + args.pred_length
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.map_size = args.map_size
        self.dataset_num = args.dataset_num
        self.data_load_step = args.data_load_step

        self.load_preprocessed_data()
        print('>> Dataset loading and analysis process are done...')

    def make_preprocessed_data(self):

        np.random.seed(0)

        print('>> making preprocessed data ..')
        target_path = self.dataset_path + '/set%d' % self.dataset_num
        file_name = os.listdir(target_path)

        f = open(target_path + '/' + file_name[0], 'rb')
        pickle_data = pickle.load(f)
        f.close()

        map_images = pickle_data[2]
        raw_data = []

        for i in range(2):

            seq_list = []
            seq_list_rel = []
            seq_list_did = []
            seq_list_pid = []

            # for training dataset
            if (i == 0):
                step_t = self.data_load_step
            # for test dataset
            elif (i == 1):
                step_t = 1

            '''
            pickle_data[0] : train files
            pickle_data[1] : test files
            '''
            cur_files = pickle_data[i]

            # for all files
            for j in range(len(cur_files)):

                # traj data from current file
                data = cur_files[j]

                # list of frame index
                frames = np.unique(data[:, 0]).tolist()

                # save as unit of frame data
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])

                # number of candidate sequences
                num_sequences = int(math.ceil(len(frames) - self.seq_length + 1))

                # for all sequences
                for idx in range(0, num_sequences + 1, step_t):

                    # current seq data of length 'self.seq_length'
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_length], axis=0)

                    # ped list in curr seq
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                    # note : ped of length 'self.seq_length' is included only !!!!
                    for _, ped_id in enumerate(peds_in_curr_seq):


                        '''
                        [_frm_idx, _id, _pos, _dataset_info]
                        '''
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                        curr_ped_dataset_id = curr_seq_data[curr_seq_data[:, 1] == ped_id, 4]

                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_length or curr_ped_seq.shape[0] != self.seq_length:
                            continue

                        # current offset sequence
                        curr_seq_rel = np.zeros((self.seq_length, 2))

                        # current sequence
                        curr_seq = np.zeros((self.seq_length, 2))

                        # offset calculation
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[1:, :] = curr_ped_seq[1:, :] - curr_ped_seq[:-1, :]

                        # add to traj list
                        curr_seq[pad_front:pad_end, :] = curr_ped_seq[:, 2:4]
                        curr_seq_rel[pad_front:pad_end, :] = rel_curr_ped_seq[:, 2:4]

                        seq_list.append(curr_seq)
                        seq_list_rel.append(curr_seq_rel)
                        seq_list_did.append(curr_ped_dataset_id)
                        seq_list_pid.append(ped_id)

            raw_data.append([seq_list, seq_list_rel, seq_list_did, seq_list_pid])

        # split train data into train and validation
        seq_list, seq_list_rel, seq_list_did, seq_list_pid = raw_data[0]
        seq_list_t, seq_list_rel_t, seq_list_did_t, seq_list_pid_t = [], [], [], []
        seq_list_v, seq_list_rel_v, seq_list_did_v, seq_list_pid_v = [], [], [], []

        for i in range(len(seq_list)):
            if (np.random.rand(1) < 0.1):
                seq_list_v.append(seq_list[i])
                seq_list_rel_v.append(seq_list_rel[i])
                seq_list_did_v.append(seq_list_did[i])
                seq_list_pid_v.append(seq_list_pid[i])
            else:
                seq_list_t.append(seq_list[i])
                seq_list_rel_t.append(seq_list_rel[i])
                seq_list_did_t.append(seq_list_did[i])
                seq_list_pid_t.append(seq_list_pid[i])

        final_data = []
        final_data.append([seq_list_t, seq_list_rel_t, seq_list_did_t, seq_list_pid_t])
        final_data.append([seq_list_v, seq_list_rel_v, seq_list_did_v, seq_list_pid_v])
        final_data.append(raw_data[1])
        final_data.append(map_images)

        # save
        save_dir = target_path + '/preprocessed_dataset_%d.cpkl' % self.dataset_num
        f = open(save_dir, "wb")
        pickle.dump((final_data), f, protocol=2)
        f.close()
        print('>> preprocessed_dataset_%d.cpkl is made .. ' % self.dataset_num)

    def load_preprocessed_data(self):

        filename = self.dataset_path + '/set%d/preprocessed_dataset_%d.cpkl' % (self.dataset_num, self.dataset_num)

        #if os.path.exists(filename):
        #    os.remove(filename)
        #    print('>> previous cpkl file is removed ..')

        if not os.path.exists(filename):
            print('>> there is no preprocessed data ..')
            self.make_preprocessed_data()

        f = open(filename, 'rb')
        raw_data = pickle.load(f)
        f.close()
        print('>> preprocessed_dataset_%d.cpkl is loaded .. ' % self.dataset_num)

        '''
        raw_data[i][0] : seq_list => < self.seq_length x 2>, ....
        raw_data[i][1] : seq_list_rel => < self.seq_length x 2>, ....
        raw_data[i][2] : seq_list_did => < self.seq_length x 1>, ....       

        '''
        self.train_data = raw_data[0]
        self.valid_data = raw_data[1]
        self.test_data = raw_data[2]
        self.map = raw_data[3]
        del raw_data

        self.num_train_scenes = len(self.train_data[0])
        self.num_batches = int(self.num_train_scenes / self.batch_size)
        self.num_val_scenes = len(self.valid_data[0])
        self.num_test_scenes = len(self.test_data[0])

        print('>> the number of valid trajs :  train(%d) / valid(%d) / test(%d)' % (
        len(self.train_data[0]), len(self.valid_data[0]), len(self.test_data[0])))

    def next_batch(self, index):

        '''
        train_data[i][0] : seq_list => < self.seq_length x 2>, ....
        train_data[i][1] : seq_list_rel => < self.seq_length x 2>, ....
        train_data[i][2] : seq_list_did => < self.seq_length x 1>, ....

        '''

        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        did_batch = []

        for i in range(len(index)):
            cur_trajs = self.train_data[0][index[i]]
            cur_offsets = self.train_data[1][index[i]]
            cur_did = self.train_data[2][index[i]]

            xo_traj = np.copy(cur_trajs[:self.obs_length, :])
            xp_traj = np.copy(cur_trajs[self.obs_length:, :])
            xo_offset = np.copy(cur_offsets[:self.obs_length, :])
            xp_offset = np.copy(cur_offsets[self.obs_length:, :])

            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            did_batch.append(cur_did)

        return xo_batch, xp_batch, xoo_batch, xpo_batch, did_batch


    def next_batch_valid(self, index):

        '''
        train_data[i][0] : seq_list => < self.seq_length x 2>, ....
        train_data[i][1] : seq_list_rel => < self.seq_length x 2>, ....
        train_data[i][2] : seq_list_did => < self.seq_length x 1>, ....

        '''

        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        did_batch = []

        for i in range(len(index)):
            cur_trajs = self.valid_data[0][index[i]]
            cur_offsets = self.valid_data[1][index[i]]
            cur_did = self.valid_data[2][index[i]]

            xo_traj = np.copy(cur_trajs[:self.obs_length, :])
            xp_traj = np.copy(cur_trajs[self.obs_length:, :])
            xo_offset = np.copy(cur_offsets[:self.obs_length, :])
            xp_offset = np.copy(cur_offsets[self.obs_length:, :])

            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            did_batch.append(cur_did)

        return xo_batch, xp_batch, xoo_batch, xpo_batch, did_batch

    def next_batch_test(self, index):

        '''
        train_data[i][0] : seq_list => < self.seq_length x 2>, ....
        train_data[i][1] : seq_list_rel => < self.seq_length x 2>, ....
        train_data[i][2] : seq_list_did => < self.seq_length x 1>, ....

        '''

        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        did_batch = []
        pid_batch = []

        for i in range(len(index)):
            cur_trajs = self.test_data[0][index[i]]
            cur_offsets = self.test_data[1][index[i]]
            cur_did = self.test_data[2][index[i]]
            cur_pid = self.test_data[3][index[i]]

            xo_traj = np.copy(cur_trajs[:self.obs_length, :])
            xp_traj = np.copy(cur_trajs[self.obs_length:, :])
            xo_offset = np.copy(cur_offsets[:self.obs_length, :])
            xp_offset = np.copy(cur_offsets[self.obs_length:, :])

            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            did_batch.append(cur_did)
            pid_batch.append(cur_pid)

        return xo_batch, xp_batch, xoo_batch, xpo_batch, did_batch, pid_batch
