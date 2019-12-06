'''
 GENERATING OCCUPANCY GRID MAP
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''

import numpy as np
import math
#import matplotlib.pyplot as plt


def rectangular_occupancy_map(xo_batch, MNP, num_peds_batch, obs_len, social_range, grid_size):

    '''
    :param xo_batch: <MNP x obs_length x 2>, ....
    :param num_peds_batch:
    :param social_range:
    :param grid_size:
    :return:
    '''


    batch_size = len(xo_batch)
    o_map_seqs = np.zeros(shape=(batch_size, obs_len, MNP, MNP, grid_size*grid_size))
    cell_size = float(social_range) / float(grid_size)

    # for all elements in a batch
    for s in range(batch_size):

        # read current sequence
        xo = xo_batch[s] # (MNP x obs_length x 2)

        # for all frames in a element
        for f in range(obs_len):

            # read current frame
            xo_peds = np.squeeze(xo[:, f, :]) # (MNP x 2)

            # for all peds in a frame
            for p in range(num_peds_batch[s]):

                # target position
                cur_y = xo_peds[p, 0].astype('float32')
                cur_x = xo_peds[p, 1].astype('float32')
                ## debug --------------------------------
                #if (p == 0):
                #    plt.plot(cur_x, cur_y, 'o')
                ## debug --------------------------------

                width_low, width_high = cur_x - social_range / 2, cur_x + social_range / 2
                height_low, height_high = cur_y - social_range / 2, cur_y + social_range / 2

                # for all neighbors
                #cur_o_map = np.zeros(shape=(MNP, grid_size, grid_size))
                for n in range(num_peds_batch[s]):

                    if (p != n):
                        other_y = xo_peds[n, 0].astype('float32')
                        other_x = xo_peds[n, 1].astype('float32')

                        if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                            continue
                        cell_x = int(np.floor((other_x - width_low) / cell_size))
                        cell_y = int(np.floor((other_y - height_low) / cell_size))

                        # debug (181204) ----------
                        cell_y = (grid_size-1) - cell_y

                        # debug --------------------------------
                        #if (p == 0):
                        #    plt.plot(other_x, other_y, '+')
                        # debug --------------------------------
                        o_map_seqs[s, f, p, n, cell_y*grid_size + cell_x] = 1

                        #cur_o_map[n, cell_y, cell_x] += 1

                #if (p==0):
                #    # debug --------------------------------
                #    print('--------- frame index %d, ped index %d' % (f, p))
                #    #print(np.flipud(np.transpose(np.sum(cur_o_map, axis=0))))
                #    print(np.sum(np.squeeze(o_map_seqs[s, f, p, :]), axis=0).reshape(grid_size, grid_size))
                #    plt.axis([cur_x-1, cur_x+1, cur_y-1, cur_y+1])
                #    plt.show()
                #    a = 0
                #    # debug --------------------------------

        #plt.plot(xo[0, :, 0], xo[0, :, 1], 'o-')
        #for p in range(1, num_peds_batch[s]):
        #    plt.plot(xo[p, :, 0], xo[p, :, 1], 'x-')
        #plt.axis([xo[0, 3, 0]-3, xo[0, 3, 0]+3, xo[0, 3, 1]-3, xo[0, 3, 1]+3])
        #plt.show()

    return o_map_seqs

def rectangular_occupancy_map_policy(xo_batch, xoo_batch, xoo_p_batch, MNP, num_peds_batch, obs_len, social_range, grid_size):

    '''
    :param xo_batch: <MNP x obs_length x 2>, ....
    :param num_peds_batch:
    :param social_range:
    :param grid_size:
    :return:
    '''


    # step 1) shift encoder output
    '''
    original sequence :    xo = [x0, x1, x2 x3]
    encoder input :      xoo  = [x0-x0, x1-x0, x2-x1, x3-x2]
                              = [0,     dx1,   dx2,   dx3]
    encoder output :    xoo_p = [dx1^,  dx2^,  dx3^,  dx4^]
        
    xoo_p_batch_shift[0, dx1^, dx2^, dx3^] = xoo_p_batch[dx1^, dx2^, dx3^]    
    
    question) dx1^ is always zero. Therefore, shifted result is [0, 0, dx2^, dx3^].
              The result before cumsum is now [x0, 0, dx2^, dx3^].
              The cumsum result is now [x0, x0, x0+dx2^, x0+dx2^+dx3^]
              The original cumsum should be [x0, x0+dx1^, x0+dx1^+dx2^, x0+dx1^+dx2^+dx3^]
    '''
    # batch_size x MNP x obs_len x 2
    xoo_p_batch_shift = np.zeros_like(xoo_p_batch)
    xoo_p_batch_shift[:, :, 1:, :] = xoo_p_batch[:, :, :-1, :]
    # debug 190111 ---
    xoo_p_batch_shift[:, :, 1, :] = xoo_batch[:, :, 1, :]

    # step 2) traj reconstruction
    xoo_p_batch_shift[:, :, 0, :] += np.array(xo_batch)[:, :, 0, :]
    xoo_p_batch_shift = np.cumsum(xoo_p_batch_shift, axis=2)

    # step 3) construct occupancy grid map
    batch_size = len(xo_batch)
    o_map_seqs = np.zeros(shape=(batch_size, obs_len, MNP, MNP, grid_size*grid_size))
    cell_size = float(social_range) / float(grid_size)

    # for all elements in a batch
    for s in range(batch_size):

        # read current sequence
        xo = xo_batch[s] # (MNP x obs_length x 2)
        xop = xoo_p_batch_shift[s]  # (MNP x obs_length x 2)

        # for all frames in a element
        for f in range(obs_len):

            # read current frame
            xo_peds = np.squeeze(xo[:, f, :]) # (MNP x 2)
            xop_peds = np.squeeze(xop[:, f, :])  # (MNP x 2)

            # for all peds in a frame
            for p in range(num_peds_batch[s][0]):

                # note : current position of target ped from policy
                # target position
                cur_y = xop_peds[p, 0].astype('float32')
                cur_x = xop_peds[p, 1].astype('float32')
                # debug --------------------------------
                #plt.plot(cur_y, cur_x, 'o')
                # debug --------------------------------

                width_low, width_high = cur_x - social_range / 2, cur_x + social_range / 2
                height_low, height_high = cur_y - social_range / 2, cur_y + social_range / 2

                # note : current position of neighbor ped from gt
                # for all neighbors
                #cur_o_map = np.zeros(shape=(MNP, grid_size, grid_size))
                for n in range(num_peds_batch[s][0]):

                    if (p != n):
                        other_y = xo_peds[n, 0].astype('float32')
                        other_x = xo_peds[n, 1].astype('float32')

                        if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                            continue
                        cell_x = int(np.floor((other_x - width_low) / cell_size))
                        cell_y = int(np.floor((other_y - height_low) / cell_size))

                        # debug (181204) ----------
                        cell_y = (grid_size-1) - cell_y

                        # debug --------------------------------
                        #plt.plot(other_y, other_x, '+')
                        # debug --------------------------------
                        o_map_seqs[s, f, p, n, cell_y*grid_size + cell_x] = 1

                        #cur_o_map[n, cell_y, cell_x] += 1

                #if (p==0):
                    # debug --------------------------------
                    #print('--------- frame index %d, ped index %d' % (f, p))
                    #print(np.flipud(np.transpose(np.sum(cur_o_map, axis=0))))
                    #plt.axis([cur_y-1, cur_y+1, cur_x-1, cur_x+1])
                    # debug --------------------------------

        #plt.plot(xo[0, :, 0], xo[0, :, 1], 'o-')
        #for p in range(1, num_peds_batch[s]):
        #    plt.plot(xo[p, :, 0], xo[p, :, 1], 'x-')
        #plt.axis([xo[0, 3, 0]-3, xo[0, 3, 0]+3, xo[0, 3, 1]-3, xo[0, 3, 1]+3])
        #plt.show()

    return o_map_seqs