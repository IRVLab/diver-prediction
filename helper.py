import numpy as np
import torch
from torch.autograd import Variable

import os
import shutil
from os import walk
import math

from model import SocialModel
from vlstm_model import VLSTMModel

#one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

#(1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)
def get_method_name(index):
    # return method name given index
    return {
        1 : 'SOCIALLSTM',
        2 : 'OBSTACLELSTM',
        3 : 'VANILLALSTM'
    }.get(index, 'SOCIALLSTM')

def get_model(index, arguments, infer = False):
    # return a model given index and arguments
    if index == 1:
        return SocialModel(arguments, infer)
    elif index == 2:
        return OLSTMModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        return SocialModel(arguments, infer)

def getCoef(outputs,pointNum):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    pointNum : 0 if working on the first point, 1 if working on second point
    '''

    if pointNum == 0:
        mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    else:
        mux, muy, sx, sy, corr = outputs[:, :, 5], outputs[:, :, 6], outputs[:, :, 7], outputs[:, :, 8], outputs[:, :, 9]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], 
                [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

def getCentroid(pos):
    centroid = Variable(torch.zeros(2))
    x_min = pos[0]
    x_max = pos[2]
    y_min = pos[1]
    y_max = pos[3]
    centroid[0] = (x_min + x_max)/2
    centroid[1] = (y_min+y_max)/2
    return centroid
    
def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    IoU = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()
        IoU = IoU.cuda()

    for tstep in range(pred_length):
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)

            if nodeID not in trueNodesPresent[tstep]:
                continue

            nodeID = look_up[nodeID]


            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]
            # if diver does not actually exist in the frame
            if np.allclose(true_pos.cpu(), np.array([0,0,0,0], dtype=float)):
                continue 
            
            trueCentroid = getCentroid(true_pos)
            predCentroid = getCentroid(pred_pos)

            error[tstep] += torch.norm(predCentroid - trueCentroid, p=2)
            IoU[tstep] += box_iou(true_pos,pred_pos)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter
            IoU[tstep] = IoU[tstep] / counter 

    return torch.mean(error), torch.mean(IoU)

def box_iou(truePos, predPos):
    """
    Helper funciton to calculate intersection over the union of two boxes a and b
     Bbox coordinates =>> {left, right, top, bottom}
    """
    groundTruth = truePos.cpu()
    predictedLoc = predPos.cpu()
    #    col min         col max        row min        row max
    a = [groundTruth[0],groundTruth[2],groundTruth[1],groundTruth[3]]
    b = [predictedLoc[0],predictedLoc[2],predictedLoc[1],predictedLoc[3]]
    w_intsec = np.maximum (0, (np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[2], b[2])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[1] - a[0])*(a[3] - a[2])
    s_b = (b[1] - b[0])*(b[3] - b[2])
    return float(s_intsec)/(s_a + s_b -s_intsec)

def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0
    IoU = 0
    # Last time-step
    tstep = pred_length - 1
    # if pred_length == 100:
    #     print("Pred pos:", ret_nodes[tstep,:,:])
    #     print("True pos:", nodes[tstep, :,:])
    #     print("assumed nodes:", assumedNodesPresent[tstep])
    #     print("true nodes:", trueNodesPresent[tstep])
    #     print("lookup:",look_up)
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)

        if nodeID not in trueNodesPresent[tstep]:
            continue

        nodeID = look_up[nodeID]

        
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        # Skip if the diver doesn't actually exist. Similar check as above
        if np.allclose(true_pos.cpu(), np.array([0,0,0,0], dtype=float)):
            continue 
        trueCentroid = getCentroid(true_pos)
        predCentroid = getCentroid(pred_pos)

        error += torch.norm(predCentroid - trueCentroid, p=2)
        IoU += box_iou(true_pos,pred_pos)
        counter += 1
        
    if counter != 0:
        error = error / counter
        IoU = IoU / counter
   
    return error, IoU

def get_normalized_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    img_normalized_centroid_error = 0
    box_normalized_centroid_error = 0
    error = 0
    counter = 0
    IoU = 0

    # Last time-step
    tstep = pred_length - 1
    
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)

        if nodeID not in trueNodesPresent[tstep]:
            continue

        nodeID = look_up[nodeID]
        
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        # Skip if the diver doesn't actually exist. Similar check as above
        if np.allclose(true_pos.cpu(), np.array([0,0,0,0], dtype=float)):
            continue 
        
        trueCentroid = getCentroid(true_pos)
        predCentroid = getCentroid(pred_pos)

        # ABSOLUTE ERROR
        error += torch.norm(predCentroid - trueCentroid, p=2)

        # IMAGE NORMALIZED ERROR
        img_dim = np.array([320,240])
        trueCentroid_imgNorm = trueCentroid / img_dim
        predCentroid_imgNorm = predCentroid / img_dim
        img_normalized_centroid_error += torch.norm(predCentroid_imgNorm - trueCentroid_imgNorm, p=2)
        
        # BOX NORMALIZED ERROR
        box_dim = np.array([true_pos[2]-true_pos[0], true_pos[3]-true_pos[1]])
        trueCentroid_boxNorm = trueCentroid / box_dim
        predCentroid_boxNorm = predCentroid / box_dim
        box_normalized_centroid_error += torch.norm(predCentroid_boxNorm - trueCentroid_boxNorm, p=2)

        IoU += box_iou(true_pos,pred_pos)
        counter += 1
        
    img_normalized_centroid_error /= counter
    box_normalized_centroid_error /= counter
    error /= counter
    IoU = IoU / counter

    return img_normalized_centroid_error, box_normalized_centroid_error, error, IoU

def getAllErrors(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    full_pred_length = ret_nodes.size()[0]
    seq_IOU = np.zeros((full_pred_length))
    seq_ImgNorm_centroid_error = np.zeros((full_pred_length))
    seq_BoxNorm_centroid_error = np.zeros((full_pred_length))
    seq_centroid_error = np.zeros((full_pred_length))
    for predLength in range(full_pred_length):
        imgNormError, boxNormError, error, IoU = get_normalized_final_error(ret_nodes[:predLength+1], nodes[:predLength+1], assumedNodesPresent[:predLength+1], trueNodesPresent[:predLength+1], look_up)
        seq_IOU[predLength] = IoU
        seq_ImgNorm_centroid_error[predLength] = imgNormError
        seq_BoxNorm_centroid_error[predLength] = boxNormError
        seq_centroid_error[predLength] = error
    return seq_ImgNorm_centroid_error, seq_BoxNorm_centroid_error, seq_centroid_error, seq_IOU

def Gaussian2DLikelihoodInference(outputs, targets, nodesPresent, pred_length, look_up):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time

    Parameters:

    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length

    numInputPoints = 2
    i = 0

    net_loss = 0

    while i < numInputPoints:
        # Extract mean, std devs and correlation
        mux, muy, sx, sy, corr = getCoef(outputs,i)

        # Compute factors
        if i == 0:
            normx = targets[:, :, 0] - mux
            normy = targets[:, :, 1] - muy
        else:
            normx = targets[:, :, 2] - mux
            normy = targets[:, :, 3] - muy

        sxsy = sx * sy

        z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
        negRho = 1 - corr**2

        # Numerator
        result = torch.exp(-z/(2*negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        # Final PDF calculation
        result = result / denom

        # Numerical stability
        epsilon = 1e-20

        result = -torch.log(torch.clamp(result, min=epsilon))
        #print(result)

        loss = 0
        counter = 0

        for framenum in range(obs_length, seq_length):
            nodeIDs = nodesPresent[framenum]
            nodeIDs = [int(nodeID) for nodeID in nodeIDs]

            for nodeID in nodeIDs:

                nodeID = look_up[nodeID]
                loss = loss + result[framenum, nodeID]
                counter = counter + 1

        if counter != 0:
            net_loss += loss / counter

        i += 1

    
    return (net_loss/numInputPoints)


def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
    i = 0
    numInputPoints = 2 #num of points for each diver - 2 here for a bounding box of a scuba diver
    net_loss = 0

    while i < numInputPoints:

        # Extract mean, std devs and correlation
        mux, muy, sx, sy, corr = getCoef(outputs,i)
        #print("received coefficients")

        # Compute factors
        if i == 0:
            normx = targets[:, :, 0] - mux
            normy = targets[:, :, 1] - muy
        else:
            normx = targets[:, :, 2] - mux
            normy = targets[:, :, 3] - muy

        sxsy = sx * sy

        z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
        negRho = 1 - corr**2

        # Numerator
        result = torch.exp(-z/(2*negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        # Final PDF calculation
        result = result / denom

        # Numerical stability
        epsilon = 1e-20

        result = -torch.log(torch.clamp(result, min=epsilon))

        loss = 0
        counter = 0

        for framenum in range(seq_length):
            #print("At frame num ",framenum)

            nodeIDs = nodesPresent[framenum]
            nodeIDs = [int(nodeID) for nodeID in nodeIDs]

            for nodeID in nodeIDs:
                #print("At node ID ",nodeID)
                nodeID = look_up[nodeID]
                loss = loss + result[framenum, nodeID]
                counter = counter + 1

        if counter != 0:
            net_loss += loss / counter
        else:
            net_loss += loss
        
        i += 1

    #print("done")
    
    return (net_loss/numInputPoints)

##################### Data related methods ######################

def remove_file_extention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]

def add_file_extention(file_name, extention):
    # add file extension (.txt) given filename

    return file_name + '.' + extention

def clear_folder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder succesfully removed: ", path)
    else:
        print("No such path: ",path)

def delete_file(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File succesfully deleted: ", file_path)
            else:    ## Show an error ##
                print("Error: %s file not found" % file_path)        
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename,e.strerror))

def get_all_file_names(path):
    # return all file names given directory
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

def unique_list(l):
  # get unique elements from list
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x

def angle_between(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    #substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    num_inputs = 4
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:num_inputs]
            vectorized_x_seq[ind, lookup_seq[ped], 0:num_inputs]  = frame[lookup_seq[ped], 0:num_inputs] - first_values_dict[ped][0:num_inputs]

    return vectorized_x_seq, first_values_dict

def translate(x_seq, PedsList_seq, lookup_seq, value):
    # translate al trajectories given x and y values
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            vectorized_x_seq[ind, lookup_seq[ped], 0:4]  = frame[lookup_seq[ped], 0:4] - value[0:4]

    return vectorized_x_seq

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    absolute_x_seq = absolute_x_seq.cuda()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:4] = frame[lookup_seq[ped], 0:4].cuda() + first_values_dict[ped][0:4].cuda()

    return absolute_x_seq


def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        #return torch.cat([qx, qy])
        return [qx, qy]

def time_lr_scheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer

def sample_validation_data(x_seq, Pedlist, grid, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index


    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0
    numInputPoints = 2 #Number of inputs for each diver

    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.seq_length, numx_seq, 4))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()

        ret_x_seq[0] = x_seq[0]

        # For the observed part of the trajectory
        for tstep in range(args.seq_length -1):
            loss = 0
            # Do a forward prop
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 4), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            i = 0
            while i < numInputPoints:
                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(out_,i)
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
                
                if i == 0:
                    ret_x_seq[tstep + 1, :, 0] = next_x
                    ret_x_seq[tstep + 1, :, 1] = next_y
                else:
                    ret_x_seq[tstep + 1, :, 2] = next_x
                    ret_x_seq[tstep + 1, :, 3] = next_y

                loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep].view(1, numx_seq, 4), [Pedlist[tstep]], look_up)
                total_loss += loss
                i += 1

    total_loss /= numInputPoints
    return ret_x_seq, total_loss / args.seq_length


def sample_validation_data_vanilla(x_seq, Pedlist, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function for vanilla method
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index

    '''

    numInputPoints = 2
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0

    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.seq_length, numx_seq, 4), volatile=True)

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()

        ret_x_seq[0] = x_seq[0]

        # For the observed part of the trajectory
        for tstep in range(args.seq_length -1):
            #print("Timestep %d of %d"%(tstep+1,args.seq_length-1))
            loss = 0
            # Do a forward prop
            #print("Calling LSTM net at timestep ",tstep)
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 4), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            i = 0
            while i < numInputPoints:
                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(out_,i)
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
                
                if i == 0:
                    ret_x_seq[tstep + 1, :, 0] = next_x
                    ret_x_seq[tstep + 1, :, 1] = next_y
                else:
                    ret_x_seq[tstep + 1, :, 2] = next_x
                    ret_x_seq[tstep + 1, :, 3] = next_y
                
                loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep + 1].view(1, numx_seq, 4), [Pedlist[tstep+1]], look_up)
                total_loss += loss
            
                i += 1

    total_loss /= numInputPoints
    return ret_x_seq, total_loss / args.seq_length


def rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq):
    # rotate sequence given angle
    origin = (0, 0)
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            point = frame[lookup_seq[ped], 0:2]
            rotated_point = rotate(origin, point, angle)
            vectorized_x_seq[ind, lookup_seq[ped], 0] = rotated_point[0]
            vectorized_x_seq[ind, lookup_seq[ped], 1] = rotated_point[1]
    return vectorized_x_seq

def getVelocityFromObsFrames(obsFrames):
    numObs, numDivers, _ = obsFrames.shape
    avgVel = np.zeros((numDivers,2), dtype=float)
    for i in range(1,numObs):
        center1 = np.array([obsFrames[i,:,2] - obsFrames[i,:,0], obsFrames[i,:,3] - obsFrames[i,:,1]])
        center2 = np.array([obsFrames[i-1,:,2] - obsFrames[i-1,:,0], obsFrames[i-1,:,3] - obsFrames[i-1,:,1]])
        delta = np.subtract(center1,center2)
        avgVel = np.add(avgVel, delta)
    return (avgVel / numObs)

def predictionToVector(predictions, uncertainties, obsLen = 5):
    """
    predictions: (obs len + pred len) x numDivers x 4 array of predicted positions of divers 
    """
    last5Predictions = predictions[-5:]
    avgBox = np.zeros_like(predictions[0])
    for i in range(5):
        avgBox = np.add(last5Predictions[i], avgBox)
    avgBox = np.divide(avgBox, 5)
    centerBox = np.array([avgBox[:,2] - avgBox[:,0], avgBox[:,3] - avgBox[:,1]])
    return centerBox
        




