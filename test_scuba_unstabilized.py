import os
import pickle
import os
import pickle
import argparse
import time
import subprocess

import torch
from torch.autograd import Variable

import numpy as np
# USE THIS TO GENERATE PROOF IMAGES, STABILIZED
#from utils_optical_flow_test_midStab import DataLoader

# USE THIS TO TRAIN A STABILIZED MODEL
#from utils_optical_flow_test import DataLoader

# USE THIS TO TRAIN AN UNSTABILIZED MODEL
from utils_test import DataLoader
import matplotlib.pyplot as plt
from helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from helper import *
from grid import getSequenceGridMask, getGridMask

def main():
    
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=50,
                        help='Predicted length of the trajectory')
    
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=49,
                        help='Epoch of model to be loaded')
    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    # drive support
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of iteration -> we are trying many times to get lowest test error derived from observed part and prediction of observed
    # part.Currently it is useless because we are using direct copy of observed part and no use of prediction.Test error will be 0.
    parser.add_argument('--iteration', type=int, default=5,
                        help='Number of iteration to create test file (smallest test errror will be selected)')
    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    #for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'

    #run sh file for folder creation
    if not os.path.isdir("log/"):
      print("Directory creation script is running...")
      subprocess.call([f_prefix+'/make_directories.sh'])

    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    print("Selected method name: ", method_name, " model name: ", model_name)

    # Save directory
    save_base = 'models/'
    save_directory = os.path.join(save_base, method_name, model_name)
    
    #plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

    result_directory = os.path.join(f_prefix, 'result/', method_name)
    plot_test_file_directory = 'test'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    saved_args.seq_length = sample_args.pred_length + sample_args.obs_length
    seq_lenght = sample_args.pred_length + sample_args.obs_length
    
    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 1, seq_lenght, forcePreProcess = True, infer=True)
    create_directories(os.path.join(result_directory, model_name), dataloader.get_all_directory_namelist())
    create_directories(plot_directory, [plot_test_file_directory])
    dataloader.reset_batch_pointer()
    
    dataset_pointer_ins = dataloader.dataset_pointer

    
    smallest_err = 100000
    smallest_err_iter_num = -1
    bestIoU = -1
    bestIoU_iter = -1
    origin = (0,0)
    reference_point = (0,1)

    submission_store = [] # store submission data points (txt)
    result_store = [] # store points for plotting

    for iteration in range(sample_args.iteration):
        # Initialize net
        net = get_model(sample_args.method, saved_args, True)

        if sample_args.use_cuda:        
            net = net.cuda()

        # Get the checkpoint path
        model_save_name = save_tar_name+str(sample_args.epoch)+'_seqLen_' + str(120) + '_unstabilized'
        checkpoint_path = os.path.join(save_directory, model_save_name+'.tar')
        print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            print("Cannot find model")
            return
        
        # For each batch
        iteration_submission = []
        iteration_result = []
        results = []
        submission = []

       
        # Variable to maintain total mean errors over the batch
        total_error = 0
        totalIoU = 0

        # Variable to store total final errors of the batch
        batch_final_error = 0
        batch_final_IoU = 0

        # Store centroid errors and IOU numbers for each batch. Will calculate mean at the end
        allIoU = np.zeros((dataloader.num_batches, sample_args.pred_length))
        allMeanErrors = np.zeros((dataloader.num_batches, sample_args.pred_length))
        allMeanErrors_imageNorm = np.zeros((dataloader.num_batches, sample_args.pred_length))
        allMeanErrors_boxNorm = np.zeros((dataloader.num_batches, sample_args.pred_length))

        fullTime = np.zeros((dataloader.num_batches))
        modelTime = np.zeros((dataloader.num_batches))

        for batch in range(dataloader.num_batches):
            start_full = time.time()
            # Get data
            x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch()
            
            # Get the sequence
            x_seq, d_seq ,numPedsList_seq, PedsList_seq = x[0], d[0], numPedsList[0], PedsList[0]

            #dense vector creation
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            #print(lookup_seq)
                
            #dataloader.clean_test_data(x_seq, target_id, sample_args.obs_length, sample_args.pred_length)
            #dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, sample_args.obs_length, sample_args.pred_length)

            
            #get processing file name and then get dimensions of file
            folder_name = dataloader.get_directory_name_with_pointer(d_seq)
            dataset_data = dataloader.get_dataset_dimension(folder_name)
            
            #will be used for error calculation
            orig_x_seq = x_seq.clone() 
            
            # if target_id not in lookup_seq:
            #     print("pedestrian with target id %d not in frame. Skipping"%target_id)
            #     continue
            #target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:4]
            
            #grid mask calculation
            if sample_args.method == 2: #obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, lookup_seq,True)
            elif  sample_args.method == 1: #social lstm   
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda,lookup_seq)

            #vectorize datapoints
            x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

            # <------------- Experimental block ---------------->
            # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)
            # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
            # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
            # grid_seq = getSequenceGridMask(x_seq[:sample_args.obs_length], dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, sample_args.use_cuda)
            # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


            if sample_args.use_cuda:
                x_seq = x_seq.cuda()

            # The sample function
            start_modelInit = time.time()
            if sample_args.method == 3: #vanilla lstm
                # Extract the observed part of the trajectories
                obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
                ret_x_seq,uncertainties = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru)

            else:
                # Extract the observed part of the trajectories
                obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length], grid_seq[:sample_args.obs_length]
                ret_x_seq,uncertainties = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru, obs_grid)
            
            end_modelEnd = time.time()
            #revert the points back to original space
            # find the last number of peds seen in obs len
            lastObservedPeds = [PedsList_seq[sample_args.obs_length-1]]*seq_lenght
            ret_x_seq = revert_seq(ret_x_seq, lastObservedPeds, lookup_seq, first_values_dict)
            end_full = time.time()

            modelTime[batch] = (end_modelEnd - start_modelInit)
            fullTime[batch] = (end_full - start_full)
            # <--------------------- Experimental inverse block ---------------------->
            # ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, target_id_values, first_values_dict)
            # ret_x_seq = rotate_traj_with_target_ped(ret_x_seq, -angle, PedsList_seq, lookup_seq)
            # ret_x_seq = translate(ret_x_seq, PedsList_seq, lookup_seq ,-target_id_values)
            
            # Record the mean and final displacement error
            obs_len = sample_args.obs_length
            pred_len = sample_args.pred_length
            meanError, IoU = get_mean_error(ret_x_seq[obs_len:(obs_len+pred_len)].data, orig_x_seq[obs_len:(obs_len+pred_len)].data, PedsList_seq[obs_len:(obs_len+pred_len)], PedsList_seq[obs_len:(obs_len+pred_len)], sample_args.use_cuda, lookup_seq)
            total_error += meanError
            totalIoU += IoU

            finalError, finalIoU = get_final_error(ret_x_seq[obs_len:(obs_len+pred_len)].data, orig_x_seq[obs_len:(obs_len+pred_len)].data, PedsList_seq[obs_len:(obs_len+pred_len)], PedsList_seq[obs_len:(obs_len+pred_len)], lookup_seq)
            batch_final_error += finalError
            batch_final_IoU += finalIoU

            # Add to all centroid error and all IOU store
            allMeanErrors_imageNorm[batch,:], allMeanErrors_boxNorm[batch,:], allMeanErrors[batch,:], allIoU[batch,:] = getAllErrors(ret_x_seq[obs_len:(obs_len+pred_len)].data, orig_x_seq[obs_len:(obs_len+pred_len)].data, PedsList_seq[obs_len:(obs_len+pred_len)], PedsList_seq[obs_len:(obs_len+pred_len)], lookup_seq)
            
            if not np.allclose(finalIoU, allIoU[batch,-1]):
                print("Discrepancy found")
                return
        
            print('Current file : ', dataloader.get_file_name(0),' Processed trajectory number : ', batch+1, 'out of', dataloader.num_batches, 'trajectories in time', end_full - start_full)


            if dataset_pointer_ins is not dataloader.dataset_pointer:
                if dataloader.dataset_pointer is not 0:
                    iteration_submission.append(submission)
                    iteration_result.append(results)

                dataset_pointer_ins = dataloader.dataset_pointer
                submission = []
                results = []

        
            for target_id in lookup_seq:
                submission.append(submission_preprocess(dataloader, ret_x_seq.data[sample_args.obs_length:, lookup_seq[target_id], :].cpu().numpy(), sample_args.pred_length, sample_args.obs_length, target_id,uncertainties[:, lookup_seq[target_id], :].numpy()))
                #if dataloader.get_file_name() == 'barbados_scuba_007_A.txt':
                    # print(submission[0])
                #print("target id %d submitting" % target_id,ret_x_seq.data[sample_args.obs_length:, lookup_seq[target_id], :].numpy())
                results.append((x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq , dataloader.get_frame_sequence(seq_lenght), target_id, sample_args.obs_length))
                
        # Calculate the average IOU and centroid errors for each prediction length
        averageIoUs = np.mean(allIoU, axis=0)
        averageImgNormCentroidError = np.mean(allMeanErrors_imageNorm, axis=0)
        averageBoxNormCentroidError = np.mean(allMeanErrors_boxNorm, axis=0)
        averageCentroidError = np.mean(allMeanErrors, axis=0)
        averageFullTime = np.mean(fullTime)
        averageModelTime = np.mean(modelTime)

        # f = open('metric_results.txt','a')
        # f.write("Results for Unstabilized "+method_name +'\n')
        # f.write("ImgNorm: %.3f\n"%(averageFullTime))
        # f.write("BoxNorm: %.3f\n\n"%(averageModelTime))
        # f.write("IOU: ")
        # f.close()
        #predLengths = np.arange(1,sample_args.pred_length + 1)

        print("IOU:", list(averageIoUs))
        print("###############")
        print("Img Norm Centroid error:", list(averageImgNormCentroidError))
        print("###############")
        print("Box Norm Centroid error:", list(averageBoxNormCentroidError))
        print("###############")
        print("Absolute Centroid error:", list(averageCentroidError))

        # IoU plot
        # plt.clf()
        # plt.plot(predLengths, averageIoUs)
        # if sample_args.method == 1: # Social-LSTM
        #     plt.title('Average IoU vs Prediction Length for Social LSTM')
        # else: # Vanilla LSTM
        #     plt.title('Average IoU vs Prediction Length for Vanilla LSTM')
        # plt.xlabel('Prediction Length in frames')
        # plt.ylabel('IoU')
        # plt.savefig('IoU_plot.png')
        

        # plt.clf()

        # # Centroid error plot
        # plt.plot(predLengths, averageCentroidError)
        # if sample_args.method == 1: # Social-LSTM
        #     plt.title('Average Centroid Error vs Prediction Length for Social LSTM')
        # else: # Vanilla LSTM
        #     plt.title('Average Centroid Error vs Prediction Length for Vanilla LSTM')
        # plt.xlabel('Prediction Length in frames')
        # plt.ylabel('Centroid Error')
        # plt.savefig('Centroid_error_plot.png')
        
        
        iteration_submission.append(submission)
        iteration_result.append(results)

        submission_store.append(iteration_submission)
        result_store.append(iteration_result)

        if total_error<smallest_err:
            print("**********************************************************")
            print('Best iteration for Euclidean error has been changed. Previous best iteration: ', smallest_err_iter_num+1, 'Error: ', smallest_err / dataloader.num_batches)
            print('New best iteration : ', iteration+1, 'Error: ',total_error / dataloader.num_batches)
            smallest_err_iter_num = iteration
            smallest_err = total_error
        
        if totalIoU > bestIoU:
            print("**********************************************************")
            print('Best iteration for IoU has been changed. Previous best iteration: ', bestIoU_iter+1, 'IoU: ', bestIoU / dataloader.num_batches)
            print('New best iteration : ', iteration+1, 'Error: ',totalIoU / dataloader.num_batches)
            bestIoU_iter = iteration
            bestIoU = totalIoU

        print('Iteration:' ,iteration+1,' Total test mean error of the model is ', total_error / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Total test final error of the model is ', batch_final_error / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Total test mean IoU error of the model is ', totalIoU / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Total test final IoU of the model is ', batch_final_IoU / dataloader.num_batches)
        #print(submission)

    print('Smallest error iteration for mean error:', smallest_err_iter_num+1)
    print('Smallest mean error ',smallest_err / dataloader.num_batches)
    print('Best IoU', bestIoU / dataloader.num_batches)

    dataloader.write_to_file(submission_store[bestIoU_iter], result_directory, prefix, model_name)
    dataloader.write_to_plot_file(result_store[bestIoU_iter], os.path.join(plot_directory, plot_test_file_directory))


def sample(x_seq, Pedlist, args, net, true_x_seq, true_Pedlist, saved_args, dimensions, dataloader, look_up, num_pedlist, is_gru, grid = None):
    '''
    The sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    true_x_seq: True positions
    true_Pedlist: The true peds present in each frame
    saved_args: Training arguments
    dimensions: The dimensions of the dataset
    target_id: ped_id number that try to predict in this sequence
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    with torch.no_grad():
        # Construct variables for hidden and cell states
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not is_gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 4))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()


        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):
            if grid is None: #vanilla lstm
               # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 4), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            else:
                # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 4), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            numInputPoints = 2 #Per diver - 2 for a diver's bounding box

            for i in range(numInputPoints):
                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(out_obs,i)
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)
                if i == 0:
                    ret_x_seq[tstep  + 1, :, 0] = next_x
                    ret_x_seq[tstep + 1, :, 1] = next_y
                else:
                    ret_x_seq[tstep + 1, :, 2] = next_x
                    ret_x_seq[tstep + 1, :, 3] = next_y


        ret_x_seq[:args.obs_length, :, :] = x_seq.clone()

        # Last seen grid
        if grid is not None: #no vanilla lstm
            prev_grid = grid[-1].clone()

        
        
        #assign last position of observed data to temp
        #temp_last_observed = ret_x_seq[args.obs_length-1].clone()
        #ret_x_seq[args.obs_length-1] = x_seq[args.obs_length-1]
        lastObservedPeds = Pedlist[args.obs_length-1]
        lastnumPeds = num_pedlist[args.obs_length-1]
        #lastnumPeds = numx_seq
        predictionUncertainties = Variable(torch.zeros(args.pred_length,np.shape(ret_x_seq)[1],10)) 
        # For each prediction, we have two points - Each of which has 5 uncertainties (means, std devs and corr)

        # For the predicted part of the trajectory
        predictionNum = 0
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length - 1):
            # Do a forward prop
            if grid is None: #vanilla lstm
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 4), hidden_states, cell_states, [lastObservedPeds], [lastnumPeds], dataloader, look_up)
            else:
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 4), [prev_grid], hidden_states, cell_states, [lastObservedPeds], [lastnumPeds], dataloader, look_up)

            numInputPoints = 2 #Per diver

            for i in range(numInputPoints):
                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = getCoef(outputs,i)
                
                # Sample from the bivariate Gaussian
                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, lastObservedPeds, look_up)
                # print("RET X ",np.shape(ret_x_seq))
                # print("PRED ",np.shape(predictionUncertainties))
                # print("MU ",np.shape(mux))

                # Store the predicted position:
                if i == 0:
                    ret_x_seq[tstep+1, :, 0] = next_x
                    ret_x_seq[tstep+1, :, 1] = next_y
                    predictionUncertainties[predictionNum, :, 0] = mux[0, :]
                    predictionUncertainties[predictionNum, :, 1] = muy[0, :]
                    predictionUncertainties[predictionNum, :, 2] = sx[0, :]
                    predictionUncertainties[predictionNum, :, 3] = sy[0, :]
                    predictionUncertainties[predictionNum, :, 4] = corr[0, :]
                    
                else:
                    ret_x_seq[tstep+1, :, 2] = next_x
                    ret_x_seq[tstep+1, :, 3] = next_y
                    predictionUncertainties[predictionNum, :, 5] = mux[0, :]
                    predictionUncertainties[predictionNum, :, 6] = muy[0, :]
                    predictionUncertainties[predictionNum, :, 7] = sx[0, :]
                    predictionUncertainties[predictionNum, :, 8] = sy[0, :]
                    predictionUncertainties[predictionNum, :, 9] = corr[0, :]

            # List of x_seq at the last time-step (assuming they exist until the end)
            #true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in lastObservedPeds]
            next_ped_list = lastObservedPeds.copy()
            converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

            if args.use_cuda:
                list_of_x_seq = list_of_x_seq.cuda()
           
            #Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq[tstep], 0, list_of_x_seq)

            if grid is not None: #no vanilla lstm
                # Compute the new grid masks with the predicted positions
                if args.method == 2: #obstacle lstm
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(lastObservedPeds),saved_args.neighborhood_size, saved_args.grid_size, True)
                elif  args.method == 1: #social lstm   
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(lastObservedPeds),saved_args.neighborhood_size, saved_args.grid_size)

                prev_grid = Variable(torch.from_numpy(prev_grid).float())
                if args.use_cuda:
                    prev_grid = prev_grid.cuda()
            
            predictionNum += 1

        #ret_x_seq[args.obs_length-1] = temp_last_observed

        return ret_x_seq, predictionUncertainties


def submission_preprocess(dataloader, ret_x_seq, pred_length, obs_length, target_id, uncertainties):
    seq_lenght = pred_length + obs_length

    #begin and end index of obs. frames in this seq.
    begin_obs = (dataloader.frame_pointer - seq_lenght)
    end_obs = (dataloader.frame_pointer - pred_length)

    # get original data for frame number and ped ids
    observed_data = dataloader.orig_data[dataloader.dataset_pointer][begin_obs:end_obs, :]
    frame_number_predicted = dataloader.get_frame_sequence(pred_length)
    #print("attaching %d to %d"%(frame_number_predicted[0], frame_number_predicted[-1]))
    ret_x_seq_c = ret_x_seq.copy()
    #ret_x_seq_c[:,[0,1]] = ret_x_seq_c[:,[1,0]] # x, y -> y, x
    repeated_id = np.repeat(target_id, pred_length) # add id
    id_integrated_prediction = np.append(repeated_id[:, None], ret_x_seq_c, axis=1)
    frame_integrated_prediction = np.append(frame_number_predicted[:,None], id_integrated_prediction, axis=1) #add frame number
    uncertainty_integrated_prediction = np.append(frame_integrated_prediction,uncertainties, axis=1)
    #result = np.append(observed_data, frame_integrated_prediction, axis = 0)

    #return result
    return uncertainty_integrated_prediction


if __name__ == '__main__':
    main()