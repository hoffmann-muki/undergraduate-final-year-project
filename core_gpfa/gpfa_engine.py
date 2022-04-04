# Run GPFA to extract trajectories

import numpy as np
from core_gpfa.fastfa import fastfa  # CHECK if import works
from core_gpfa.exact_inference_with_LL import exact_inference_with_LL
from core_gpfa.em import em
from Seq_Data_Class import Param_Class
import scipy.io as sio
from core_gpfa.init_sm_hyper import init_sm_hyper, init_sm_hyper_v2
from core_gpfa.cosmoother_gpfa_viaOrth_fast import cosmoother_gpfa_viaOrth_fast


def save_results(fname, result):
    # Saving a dict with keys: 'LL', 'params', 'seq_train' and 'seq_test'
    sio.savemat(fname, mdict=result, format='5')


def gpfa_engine(seq_train, seq_test, fname, x_dim, bin_width, param_cov_type='rbf', param_distance='Euclidean', param_Q=3, start_tau=100, start_eps=1e-3, min_var_frac=0.01):
    # seq_train - array with 3 tuples of -
    #   trialId (1 x 1)   - unique trial identifier
    #   y (# neurons x T) - neural data
    #   T (1 x 1)         - number of timesteps
    # start_tau - GP timescale initialization in msec
    # start_eps - GP noise variance initialization

    # Initialize state model parameters
    # Initialize GP params
    param_eps = start_eps * np.ones((x_dim,))       # GP noise variance
    initialize_hyperparam = True

    y_all = np.concatenate([trial.y for trial in seq_train], 1)
    
    # Initialize observation model parameters
    # Run FA to initialize parameters
    print('\nRunning FA model for initialization\n')
    fa_params_L, fa_params_Ph, fa_params_d, _ = fastfa(y_all, x_dim)

    param_d = fa_params_d
    param_C = fa_params_L
    param_R = np.diag(fa_params_Ph)

    # Define parameter constraints
    param_notes_learnKernelParams = True
    param_notes_learnGPNoise = False
    param_notes_RforceDiagonal = True

    # Choose the distance/metric measure
    param_distance = param_distance

    # Fit model parameters
    if not (param_cov_type == 'lin' or param_cov_type == 'poly' or param_cov_type == 'nn'):
        print('\nFitting GPFA model with %s kernel, using %s distance\n' % (param_cov_type, param_distance))
    else:
        print('\nFitting GPFA model with %s kernel\n' % param_cov_type)
    
    if param_cov_type != 'sm':

        if param_cov_type == 'rbf':
            if param_distance == 'Root Manhattan':
                param_gamma = bin_width * np.ones((x_dim,))
            elif param_distance == 'Lee' or param_distance == 'Canberra' or param_distance == 'Discrete':
                param_gamma = np.zeros((x_dim,)) # not used
            else:
                param_gamma = bin_width**2 * np.ones((x_dim,)) # Euclidean / Manhattan

        elif param_cov_type == 'pw':
            if param_distance == 'Root Manhattan':
                param_gamma = np.sqrt(bin_width) * np.ones((x_dim,))
            elif param_distance == 'Canberra' or param_distance == 'Discrete':
                param_gamma = np.zeros((x_dim,)) # not used
            elif param_distance == 'Lee':
                param_gamma = 0.125 * np.ones((x_dim,)) # only used to control frequency of oscillations of sinc function
            else:
                param_gamma = bin_width * np.ones((x_dim,)) # Manhattan / Euclidean

        elif param_cov_type == 'im':
            if param_distance == 'Root Manhattan':
                param_gamma = bin_width * np.ones((x_dim,))
            elif param_distance == 'Lee' or param_distance == 'Canberra' or param_distance == 'Discrete':
                param_gamma = np.zeros((x_dim,)) # not used
            else:
                param_gamma = bin_width**2 * np.ones((x_dim,)) # Euclidean / Manhattan
        
        else:
            param_gamma = np.zeros((x_dim,)) # not used
        
        current_params = Param_Class(param_cov_type, param_gamma, param_eps, param_d, param_C, param_R, param_notes_learnKernelParams, param_notes_learnGPNoise, param_notes_RforceDiagonal, param_distance=param_distance)

        (est_params, seq_train, LLcut, iter_time) = em(current_params, seq_train, min_var_frac)
        (seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params, getLL=True)

    elif param_cov_type == 'p':

        param_p = bin_width  # period of function
        param_lp = 0.5  # lengthscale of periodic function
        param_gamma = (2 * np.pi/param_p) * bin_width * np.ones((x_dim,)) # only used in hyperparameter learning

        current_params = Param_Class(param_cov_type, param_gamma, param_eps, param_d, param_C, param_R, param_notes_learnKernelParams, param_notes_learnGPNoise, param_notes_RforceDiagonal, param_p=param_p, param_lp=param_lp, param_distance=param_distance)
        
        (est_params, seq_train, LLcut, iter_time) = em(current_params, seq_train, min_var_frac)
        (seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params, getLL=True)

    elif param_cov_type == 'sm':

        flag = True
        tryNum = 1

        # Try running EM / inference with different random initializations when using SM kernel
        # (found that you can get errors in logdet from both EM and inference - linalg error, cholesky decomposition of non PSD matrix)
        # You exit the while loop once you find an initialization for which
        while flag:

            param_gamma = []
            for i in range(x_dim):
                weights = np.ones(param_Q).tolist()
                weights = weights / np.sum(weights)
                weights = weights.tolist()
                mu = np.random.uniform(0, 1, param_Q).tolist()
                vs = np.random.uniform(0, 1, param_Q).tolist()
                param_gamma.append(weights + mu + vs)

            current_params = Param_Class(param_cov_type, param_gamma, param_eps, param_d, param_C, param_R, param_notes_learnKernelParams, param_notes_learnGPNoise, param_notes_RforceDiagonal, param_q=param_Q, param_distance=param_distance)

            try:

                print("\n Attempt %d of SM fitting for xdim %d \n" % (tryNum, x_dim))
                if initialize_hyperparam:

                    print('\nRunning E-step for initializing hyperparameters for SM\n')
                    (seq_train, _) = exact_inference_with_LL(seq_train, current_params, getLL=False)
                    init_gamma = np.zeros((len(seq_train), current_params.Q*3))
                    # Calculate gamma for each latent dimension and each trial
                    for d in range(x_dim):
                        for i in range(len(seq_train)):
                            init_train_x = np.arange(seq_train[i].T).reshape((seq_train[i].T, 1))
                            init_train_y = seq_train[i].xsm[d, :].T
                            hyper_params = init_sm_hyper(x=init_train_x, y=init_train_y, Q=param_Q)
                            # hyper_params = init_sm_hyper_v2(train_x=init_train_x, train_y=init_train_y, num_mixtures=param_Q)
                            init_gamma[i, :] = hyper_params

                        # Initialize with mean
                        current_params.gamma[d] = np.mean(
                            init_gamma, axis=0).tolist()

                # Attempt EM and inference
                (est_params, seq_train, LLcut, iter_time) = em(current_params, seq_train, min_var_frac)
                (seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params, getLL=True)
                print("\n Attempt %d succeeded!" % (tryNum))
                flag = False
            except Exception as e:
                print('Error:', e)
                tryNum += 1

    # Assess generalization performance
    # TODO
    LLtest = np.nan
    leave_one_out = []
    if len(seq_test) > 0:
        if est_params.RforceDiagonal:
            leave_one_out = cosmoother_gpfa_viaOrth_fast(seq_test, est_params, np.arange(x_dim))
        (_, LLtest) = exact_inference_with_LL(seq_test, est_params, getLL=True)

    result = dict({'LLtrain': LLtrain, 'LLtest': LLtest, 'params': est_params, 'seq_train': seq_train,
    'seq_test': seq_test, 'bin_width': bin_width, 'leave_one_out': leave_one_out})

    # Save results
    save_results(fname, result)

    return result
