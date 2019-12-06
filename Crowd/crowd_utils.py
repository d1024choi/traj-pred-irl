'''
 CROWD DATA LOADER BASED ON KITTI DATA LOADER
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06

dataset number | leave dataset name
------------------------------------
       0       |        eth
       1       |        hotel
       2       |        univ
       3       |        zara1
       4       |        zara2

'''

import pickle
from crowd_functions import *
from crowd_functions import _min, _max


class DataLoader:

    def __init__(self, args):

        self.dataset_num = args.dataset_num
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.seq_length = args.obs_length + args.pred_length
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.social_range = args.social_range
        self.social_grid_size = args.grid_size
        self.max_num_peds = args.max_num_peds
        self.min_ped = 1
        self.data_load_step = args.data_load_step

        if (self.dataset_num == 0):
            self.dataset_path = args.dataset_path + 'eth/'
        elif (self.dataset_num == 1):
            self.dataset_path = args.dataset_path + 'hotel/'
        elif (self.dataset_num == 2):
            self.dataset_path = args.dataset_path + 'univ/'
        elif (self.dataset_num == 3):
            self.dataset_path = args.dataset_path + 'zara1/'
        elif (self.dataset_num == 4):
            self.dataset_path = args.dataset_path + 'zara2/'


        self.load_preprocessed_data()


    def make_preprocessed_data(self):

        print('>> making preprocessed data ..')
        dataset_paths = []
        dataset_paths.append(self.dataset_path + 'train')
        dataset_paths.append(self.dataset_path + 'val')
        dataset_paths.append(self.dataset_path + 'test')

        raw_data = []
        # note : for the current folder
        for p in range(len(dataset_paths)):

            '''
            p == 0 -> train folder
            p == 1 -> val folder
            p == 2 -> test folder
            
            '''

            # all files in the current folder
            all_files = os.listdir(dataset_paths[p])
            all_files = [os.path.join(dataset_paths[p], _path) for _path in all_files]

            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            loss_mask_list = []

            # add more info
            peds_id_list = []

            # note : define data load step
            if (p == 0):
                step_t = self.data_load_step
                MNP = self.max_num_peds
            elif (p == 1):
                step_t = self.data_load_step
                MNP = self.max_num_peds
            else:
                step_t = 1
                MNP = 80

            # note : for all files in the current folder
            for path in all_files:

                # read current file data
                '''
                data : <num_frames x 4>
                data[i, 0] : frame index
                data[i, 1] : ped index
                data[i, 2] : x position
                data[i, 3] : y position
                '''
                data = read_file(path, delim='\t')

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

                    # current offset sequence
                    curr_seq_rel = np.zeros((MNP, self.seq_length, 2))

                    # current sequence
                    curr_seq = np.zeros((MNP, self.seq_length, 2))

                    # current loss mask
                    curr_loss_mask = np.zeros((MNP, 1))

                    # pad id list
                    curr_ped_ids = np.zeros((MNP, 1))

                    # note : ped of length 'self.seq_length' is included only !!!!
                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_length or num_peds_considered == (MNP-1):
                            continue

                        # offset calculation
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[1:, :] = curr_ped_seq[1:, :] - curr_ped_seq[:-1, :]

                        # add to traj list
                        _idx = num_peds_considered
                        curr_seq[_idx, pad_front:pad_end, :] = curr_ped_seq[:, 2:4]
                        curr_seq_rel[_idx, pad_front:pad_end, :] = rel_curr_ped_seq[:, 2:4]

                        # check this is true ped
                        curr_loss_mask[_idx, 0] = 1

                        # cur ped id
                        curr_ped_ids[_idx, 0] = ped_id

                        # incread valid ped number
                        num_peds_considered += 1

                    if num_peds_considered > (self.min_ped-1):
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask)
                        seq_list.append(curr_seq)
                        seq_list_rel.append(curr_seq_rel)
                        peds_id_list.append(curr_ped_ids)

            raw_data.append([num_peds_in_seq, seq_list, seq_list_rel, loss_mask_list, peds_id_list])

        # save
        save_dir = self.dataset_path + 'preprocessed_dataset_%d.cpkl' % self.dataset_num
        f = open(save_dir, "wb")
        pickle.dump((raw_data), f, protocol=2)
        f.close()
        print('>> preprocessed_dataset_%d.cpkl is made .. ' % self.dataset_num)

    def load_preprocessed_data(self):

        filename = self.dataset_path + 'preprocessed_dataset_%d.cpkl' % self.dataset_num
        if os.path.exists(filename):
            os.remove(filename)
            print('>> previous cpkl file is removed ..')
        
        if not os.path.exists(filename):
            print('>> there is no preprocessed data ..')
            self.make_preprocessed_data()

        f = open(filename, 'rb')
        raw_data = pickle.load(f)
        f.close()
        print('>> preprocessed_dataset_%d.cpkl is loaded .. ' % self.dataset_num)

        '''
        raw_data[i][0] : num_peds_in_seq 
        raw_data[i][1] : seq_list => < self.max_num_peds x self.seq_length x 2>, ....
        raw_data[i][2] : seq_list_rel => < self.max_num_peds x self.seq_length x 2>, ....
        raw_data[i][3] : loss_mask_list       
        
        '''

        self.train_data = raw_data[0]
        self.valid_data = raw_data[1]
        self.test_data = raw_data[2]
        del raw_data

        self.num_train_scenes = len(self.train_data[0])
        self.num_batches = int(self.num_train_scenes / self.batch_size)
        self.num_val_scenes = len(self.valid_data[0])
        self.num_test_scenes = len(self.test_data[0])

        print('>> the number of valid trajs :  train(%d) / valid(%d) / test(%d)' % (np.sum(self.train_data[0]), np.sum(self.valid_data[0]), np.sum(self.test_data[0]) ) )

    def next_batch(self, index):

        '''
        train_data[0][index] : num_peds_in_seq, integer
        train_data[1][index] : trajs, (100, self.seq_length, 2)
        train_data[2][index] : offsets, (100, self.seq_length, 2)
        train_data[3][index] : loss mask, (100, 1)
        '''


        np_batch = []
        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        xm_batch = []

        for i in range(len(index)):
            num_peds_in_seq = self.train_data[0][index[i]]
            cur_trajs = self.train_data[1][index[i]]
            cur_offsets = self.train_data[2][index[i]]
            cur_loss_mask = self.train_data[3][index[i]]

            xo_traj = np.copy(np.array(cur_trajs[:, :self.obs_length, :]))
            xp_traj = np.copy(np.array(cur_trajs[:, self.obs_length:, :]))
            xo_offset = np.copy(np.array(cur_offsets[:, :self.obs_length, :]))
            xp_offset = np.copy(np.array(cur_offsets[:, self.obs_length:, :]))
            x_mask = np.copy(np.array(cur_loss_mask))

            np_batch.append(num_peds_in_seq)
            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            xm_batch.append(x_mask)

        return np_batch, xo_batch, xp_batch, xoo_batch, xpo_batch, xm_batch

    def next_batch_valid(self, index):

        '''
        train_data[0][index] : num_peds_in_seq, integer
        train_data[1][index] : trajs, (100, self.seq_length, 2)
        train_data[2][index] : offsets, (100, self.seq_length, 2)
        train_data[3][index] : loss mask, (100, 1)
        '''

        np_batch = []
        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        xm_batch = []

        for i in range(len(index)):
            num_peds_in_seq = self.valid_data[0][index[i]]
            cur_trajs = self.valid_data[1][index[i]]
            cur_offsets = self.valid_data[2][index[i]]
            cur_loss_mask = self.valid_data[3][index[i]]

            xo_traj = np.copy(np.array(cur_trajs[:, :self.obs_length, :]))
            xp_traj = np.copy(np.array(cur_trajs[:, self.obs_length:, :]))
            xo_offset = np.copy(np.array(cur_offsets[:, :self.obs_length, :]))
            xp_offset = np.copy(np.array(cur_offsets[:, self.obs_length:, :]))
            x_mask = np.copy(np.array(cur_loss_mask))

            np_batch.append(num_peds_in_seq)
            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            xm_batch.append(x_mask)

        return np_batch, xo_batch, xp_batch, xoo_batch, xpo_batch, xm_batch

    def next_batch_test(self, index):

        '''
        train_data[0][index] : num_peds_in_seq, integer
        train_data[1][index] : trajs, (100, self.seq_length, 2)
        train_data[2][index] : offsets, (100, self.seq_length, 2)
        train_data[3][index] : loss mask, (100, 1)
        '''

        np_batch = []
        xo_batch = []
        xp_batch = []
        xoo_batch = []
        xpo_batch = []
        xm_batch = []
        pid_batch = []

        for i in range(len(index)):
            num_peds_in_seq = self.test_data[0][index[i]]
            cur_trajs = self.test_data[1][index[i]]
            cur_offsets = self.test_data[2][index[i]]
            cur_loss_mask = self.test_data[3][index[i]]
            cur_ped_id_list = self.test_data[4][index[i]]

            xo_traj = np.copy(np.array(cur_trajs[:, :self.obs_length, :]))
            xp_traj = np.copy(np.array(cur_trajs[:, self.obs_length:, :]))
            xo_offset = np.copy(np.array(cur_offsets[:, :self.obs_length, :]))
            xp_offset = np.copy(np.array(cur_offsets[:, self.obs_length:, :]))
            x_mask = np.copy(np.array(cur_loss_mask))
            p_id = np.copy(np.array(cur_ped_id_list))

            np_batch.append(num_peds_in_seq)
            xo_batch.append(xo_traj)
            xp_batch.append(xp_traj)
            xoo_batch.append(xo_offset)
            xpo_batch.append(xp_offset)
            xm_batch.append(x_mask)
            pid_batch.append(p_id)

            print('[scene number %d] num peds %d' %(index[i], num_peds_in_seq))

        return np_batch, xo_batch, xp_batch, xoo_batch, xpo_batch, xm_batch, pid_batch

    def data_smoothing(self, data):

        # consider all the peds
        pedList = np.unique(data[:, 1])

        frmList = np.unique(data[:, 0])

        # maximum frm idx
        frm_max = 0


        overall_trajs = []
        for _, k in enumerate(pedList):

            # extract cur_ped data
            '''
            data[:, 0] : frame
            data[:, 1] : ped id
            data[:, 2] : x
            data[:, 3] : y
            '''
            ped_data = data[data[:, 1] == k]
            ped_data_smooth = np.copy(ped_data)

            data_len = ped_data.shape[0]
            if (data_len > 2):
                for i in range(1, data_len-1):
                    avg_x = np.mean(ped_data[i - 1: i + 2, 2])
                    avg_y = np.mean(ped_data[i - 1: i + 2, 3])
                    ped_data_smooth[i, 2] = avg_x
                    ped_data_smooth[i, 3] = avg_y

            #plt.plot(ped_data[:, 2], ped_data[:, 3], 'o-')
            #plt.plot(ped_data_smooth[:, 2], ped_data_smooth[:, 3], 'x-')
            #plt.show()

            # final data
            overall_trajs.append(ped_data_smooth)


        # ------------------------------
        # re-form data

        # for all frame indices
        counter = 0
        for _, fidx in enumerate(frmList):

            # for all peds
            overall_frame_data = []
            for k in range(len(overall_trajs)):

                # current traj
                cur_traj = overall_trajs[k]

                # current frame data
                cur_frame_data = cur_traj[cur_traj[:, 0] == fidx]

                # if it is not empty
                if (len(cur_frame_data) > 0):
                    overall_frame_data.append(cur_frame_data.reshape(1, 4))

            # if it is not empty
            if (len(overall_frame_data) > 0):
                counter += 1
                if (counter == 1):
                    reform_data = np.copy(np.array(overall_frame_data))
                else:
                    reform_data = np.concatenate([reform_data, overall_frame_data], axis=0)

        return np.squeeze(reform_data)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def Calc_dist_matrix(pedList, cur_seq_data):

    dist_mat = np.zeros(shape=(len(pedList), len(pedList)))

    for _, ped_id in enumerate(pedList):
        cur_ped_seq = cur_seq_data[cur_seq_data[:, 1] == ped_id, 2:4]
        cur_ped_pos = np.mean(cur_ped_seq, axis=0)
        for __, ped_id_n in enumerate(pedList):
            cur_ped_seq_n = cur_seq_data[cur_seq_data[:, 1] == ped_id_n, 2:4]
            cur_ped_pos_n = np.mean(cur_ped_seq_n, axis=0)

            dist = np.mean((cur_ped_pos - cur_ped_pos_n)**2)

            if (ped_id == ped_id_n):
                dist_mat[_, __] = 10000
            else:
                dist_mat[_, __] = dist

    return dist_mat
