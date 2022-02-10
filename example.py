# Example script to run methods on sample data

# Code modified from the version by Byron Yu byronyu@stanford.edu, John Cunningham jcunnin@stanford.edu

from extract_traj import extract_traj, mean_squared_error, goodness_of_fit_rsquared, getPredErrorVsDim
from data_simulator import load_data, load_real_world_data
import numpy as np
from core_gpfa.postprocess import postprocess
from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')
# plt.switch_backend('agg')

# set random seed for reproducibility
# np.random.seed(1)

RUN_ID = 1
OUTPUT_DIR = './output/'+str(RUN_ID)+'/'

# x_dim = 8 for latent dimension of 'rbf'
# x_dim = 2 for latent dimension of 'sm'
x_dim = 5
method = 'gpfa'
param_cov_type = 'rbf'  # type of kernel
param_Q = 2  # number of mixtures for SM
num_folds = 3  # change to n>=2 for n-fold cross-validation
kern_SD = 30

INPUT_FILE = './input/fake_data_{}.mat'.format('sm')
# INPUT_FILE = '../em_input_new.mat'
# INPUT_FILE = '../dataForRoman_sort.mat'
# INPUT_FILE = '../fake_data2_w_genparams.mat' # '../em_input_new.mat', '../fake_data2_w_genparams.mat', '../fake_data_w_genparams.mat'

# Load data
dat = load_real_world_data('dataset/Movie2Exp2.mat', movie=True)
# dat = load_data(INPUT_FILE)

result = extract_traj(output_dir=OUTPUT_DIR, data=dat, method=method, x_dim=x_dim,
                      param_cov_type=param_cov_type, param_Q=param_Q, num_folds=num_folds)

# Extract trajectories for different dimensionalities
# dims = [2, 5, 8]
# for x_dim in dims:
#     result = extract_traj(output_dir=OUTPUT_DIR, data=dat, method=method, x_dim=x_dim,\
#                             param_cov_type=param_cov_type, param_Q = param_Q, num_folds = num_folds)

# Get leave-one-out prediction (see Yu et al., 2009 for details on GPFA reduced)
# gpfa_errs, gpfa_reduced_errs = getPredErrorVsDim(OUTPUT_DIR, method, param_cov_type, num_folds, dims)

# # Plotting can be done as follows:
# plt.plot(dims, gpfa_errs, '--k')
# plt.plot(np.arange(1,gpfa_reduced_errs.size+1),gpfa_reduced_errs)
# plt.xlabel('State dimensionality')
# plt.ylabel('Prediction error')

# Orthonormalize trajectories
# Returns results for the last run cross-validation fold, if enabled
(est_params, seq_train, seq_test) = postprocess(
    result['params'], result['seq_train'], result['seq_test'], method)

print("LL for training: %.3f, for testing: %.3f, method: %s, x_dim:%d, param_cov_type:%s, param_Q:%d"
      % (result['LLtrain'], result['LLtest'], method, x_dim, param_cov_type, param_Q))

# Output filenames for plots
output_file = OUTPUT_DIR+"/"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type

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
