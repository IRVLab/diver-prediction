import numpy as np
import torch
import itertools
from torch.autograd import Variable


def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy = False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of occupancy map

    '''
    mnp = num_person

    width, height = dimensions[0], dimensions[1]
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    frame_np =  frame.data.numpy()

    width_bound, height_bound = neighborhood_size, neighborhood_size # in pixel space 
    #print("weight_bound: ", width_bound, "height_bound: ", height_bound)

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_xmin, current_ymin, current_xmax, current_ymax = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1], frame_np[real_frame_index, 2], frame_np[real_frame_index, 3]
        if current_xmin == 0 and current_ymin == 0 and current_xmax == 0 and current_ymax == 0:
            continue
        # Set the cell co-ordinates of the diver as the centroid of its bounding box
        current_x = (current_xmin + current_xmax) / 2
        current_y = (current_ymin + current_ymax) / 2

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2
        current_diagonal = np.linalg.norm([current_xmax-current_xmin, current_ymax-current_ymin]) 

        other_xmin, other_ymin, other_xmax, other_ymax = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1], frame_np[other_real_frame_index, 2], frame_np[other_real_frame_index, 3]
        other_x = (other_xmin + other_xmax) / 2
        other_y = (other_ymin + other_ymax) / 2
        other_diagonal = np.linalg.norm([other_xmax-other_xmin, other_ymax-other_ymin])
        
        # If other diver is not in bounds, or if the difference of length of diagonals is greater than a threshold:
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low) or abs(current_diagonal-other_diagonal) > 10 :
                # Ped not in surrounding, so binary mask should be zero
                #print("not surrounding")
                continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, pedlist_seq, neighborhood_size, grid_size, using_cuda, lookup_seq, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []
    #numDivers = np.shape(sequence)[1]

    for i in range(sl):
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size, grid_size, is_occupancy)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask
