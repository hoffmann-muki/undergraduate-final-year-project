# Example script to run methods on sample data

# Code modified from the version by Byron Yu byronyu@stanford.edu, John Cunningham jcunnin@stanford.edu

from extract_traj import extract_traj, mean_squared_error, goodness_of_fit_rsquared, getPredErrorVsDim
from data_simulator import load_data, load_real_world_data
import numpy as np
import scipy.io as sio
from core_gpfa.postprocess import postprocess
from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error
import matplotlib.pyplot as plt


RUN_ID = 1
OUTPUT_DIR = './output/'+str(RUN_ID)+'/'


if __name__ == '__main__':

    method = 'gpfa'
    param_Q = 2  # number of mixtures for SM
    param_distance = ''
    param_cov_type = input('Choose a kernel:\nrbf\trq\tpw\tim\tsm\tp\tlp\tcos\tlin\tpoly\tnn\tcirc\tlogit\n>>> ').strip() or 'rbf'
    if param_cov_type == 'rbf':
        param_distance = input('Choose a metric:\nEuclidean\tRoot Manhattan\t\tLee\tCanberra\tDiscrete\tEuclidean -- varying timescales\n>>> ').strip() or 'Euclidean'
    elif param_cov_type == 'rq' or param_cov_type == 'pw' or param_cov_type == 'im':
        param_distance = input('Choose a metric:\nEuclidean\tRoot Manhattan\t\tLee\tCanberra\tDiscrete\n>>> ').strip() or 'Euclidean'
    elif param_cov_type == 'p':
        param_distance = input('Choose a metric:\nEuclidean\tLee\tDiscrete\n>>> ').strip() or 'Euclidean'
    elif param_cov_type == 'sm' or param_cov_type == 'lp' or param_cov_type == 'cos':
        param_distance = 'Euclidean'
    # x_dim = 8 for latent dimension of 'rbf' on synthetic data
    # x_dim = 2 for latent dimension of 'sm' on synthetic data
    x_dim = int(input('Choose the size of the latent dimension (a positive integer):\n>>> ').strip() or 4)
    num_folds = int(input('Choose the number of cross-validation folds:\n0 for no cross-validation, n >= 2 for n-fold cross-validation\n>>> ').strip() or 0)
    bin_width = int(input('Choose the bin width for the spike counts (a positive integer, preferably a multiple of 10):\n>>> ').strip() or 20)
    is_real_world_data = int(input('Choose the kind of data:\n0 for synthetic\t\t1 for real world\n>>> ') or 0)
    if not is_real_world_data:
        INPUT_FILE = './input/fake_data_{}.mat'.format('sm')
        data = load_data(INPUT_FILE, bin_width=1) # for synthetic data
    else:
        filename = input('Choose the dataset file:\nMovie1Exp1.mat\tMovie1Exp2.mat\tMovie2Exp1\tMovie2Exp2.mat\tNaturalImages1.mat\tNaturalImages2.mat\tShifts1.mat\tShifts2.mat\tGratings.mat\n>>> ').strip()
        dataset_format = int(input('Choose the dataset format:\n1 for Movie\t2 for Natural Images OR Gratings\t3 for Shifted Natural Images\n>>> ').strip())
        if dataset_format == 1:
            data = load_real_world_data('dataset/' + filename, bin_width=bin_width, movie=True)
        elif dataset_format == 2:
            n_stimuli = sio.loadmat('dataset/' + filename)['n_stimuli'][0][0]
            stimulus_index = int(input('Choose the stimulus index (an integer from 0 to %d):\n>>> ' % (n_stimuli-1)) or 0)
            data = load_real_world_data('dataset/' + filename, bin_width=bin_width, natural_or_gratings=True, stimulus_index=stimulus_index)
        elif dataset_format == 3:
            n_shifts = sio.loadmat('dataset/' + filename)['n_shifts'][0][0]
            n_images = sio.loadmat('dataset/' + filename)['n_images'][0][0]
            shift_index = int(input('Choose the shift index (an integer from 0 to %d):\n>>> ' % (n_shifts-1)) or 0)
            image_index = int(input('Choose the image index (an integer from 0 to %d):\n>>> ' % (n_images-1)) or 0)
            data = load_real_world_data('dataset/' + filename, bin_width=bin_width, shifted=True, shift_index=shift_index, image_index=image_index)
    
    result = extract_traj(output_dir=OUTPUT_DIR, bin_width=bin_width, data=data, method=method, x_dim=x_dim, param_cov_type=param_cov_type, param_distance=param_distance, param_Q=param_Q, num_folds=num_folds)
    
    # Orthonormalize trajectories
    # Returns results for the last run cross-validation fold, if enabled
    (est_params, seq_train, seq_test) = postprocess(result['params'], result['seq_train'], result['seq_test'], method)

    print("LL for training: %.3f, for testing: %.3f, method: %s, x_dim:%d, param_cov_type:%s, param_Q:%d"
        % (result['LLtrain'], result['LLtest'], method, x_dim, param_cov_type, param_Q))

    # Output filenames for plots
    output_file = OUTPUT_DIR+"/"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type+"_dist_"+param_distance

    # Plot trajectories in 3D space
    if x_dim >= 3:
        plot_3d(seq_train, 'x_orth', dims_to_plot=[0, 1, 2], output_file=output_file)

    # Plot each dimension of trajectory
    # plot_1d(seq_train, 'x_sm', result['bin_width'], output_file=output_file)
    plot_1d(seq_train, 'x_orth', result['bin_width'], output_file=output_file)

    # Plot all figures
    plt.show()

    # Cross-validation to find optimal state dimensionality
    # TODO
