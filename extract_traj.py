# Extracting trajectory from given data

import os
from core_gpfa.gpfa_engine import gpfa_engine
from core_gpfa.make_K_big import make_K_big
import numpy as np
from sklearn import linear_model
from scipy import sparse
from copy import deepcopy
import scipy.io as sio

# Makes a matrix full rank
def make_full_rank(M):
    full_rank = np.linalg.matrix_rank(M)
    nc = M.shape[1]
    drop_list = []
    for j in range(nc):
        P = np.delete(M, [j], 1)
        r = np.linalg.matrix_rank(P)
        if r == full_rank:
            drop_list.append(j)
    return np.delete(M, drop_list, 1)


# R^2 between actual and predicted latent trajectories
def goodness_of_fit_rsquared(seq, x_dim, xspec='xsm'):
    r2_trials = np.zeros((len(seq), x_dim))
    for n in range(len(seq)):
        
        # Dimension of predicted latent states not equal to true latent dimensions 
        if seq[n].x.shape[0] != x_dim:
            print("True and predicted latent state dimensions do not match")
            return np.full(x_dim, np.nan)

        T = seq[n].T
        pred_latent_traj = getattr(seq[n], xspec)

        for d in range(x_dim):
            act_latent_traj = seq[n].x[d,:]

            # R^2 from linear regression
            linear_reg = linear_model.LinearRegression()
            linear_reg.fit(X=pred_latent_traj.T, y=act_latent_traj.T)
            r2_value = linear_reg.score(X=pred_latent_traj.T, y=act_latent_traj.T)
            r2_trials[n,d] = r2_value

    print("r2_trials", r2_trials)
    mean_r2_trials = np.mean(r2_trials, axis=0)

    return mean_r2_trials


# Mean squared error between actual and predicted latent trajectories
def mean_squared_error(seq, xspec='xsm'):
    error_trials = np.zeros(len(seq))
    for n in range(len(seq)):
        x_dim = (seq[n].xsm).shape[0]
        
        # Dimension of predicted latent states not equal to true latent dimensions 
        if seq[n].x.shape[0] != x_dim:
            print("True and predicted latent state dimensions do not match")
            return np.inf

        T = seq[n].T
        pred = getattr(seq[n], xspec)
        # Frobenius norm
        error = np.sum(np.power(pred - seq[n].x, 2))
        # Normalize by x_dim*T
        error = error * 1.0 / (x_dim * T)
        error_trials[n] = error
    print("error_trials", error_trials)
    mean_error_trials = np.mean(error_trials)

    return mean_error_trials


def getPredErrorVsDim(OUTPUT_DIR, method, param_cov_type, num_folds, dims):
    
    # Compute the leave-one-out prediction error for each dimension separately

    # Loop over dimensions
    dim_errs = []
    for x_dim in dims:
        # Loop over folds
        fold_errs = np.zeros((num_folds, x_dim))
        sse = []
        for fold in np.arange(1, num_folds+1):

            # Define output file to load
            file = OUTPUT_DIR+method+'_xdim_'+str(x_dim)+"_cov_"+param_cov_type+"_cv"+str(fold)+".mat"
            curr_file = sio.loadmat(file)

            # Observations for the test set
            YtestRaw = np.concatenate([curr_file['seq_test'].flatten()[i][0][0][3] \
            for i in range(curr_file['seq_test'].flatten().size)], 1)
            errs = []
            # Leave one out predictions across all trials within given dimension
            for p in range(x_dim):
                Ycs = np.concatenate([curr_file['leave_one_out'].flatten()[i]['dim'+str(p)][0][0] \
                for i in range(curr_file['leave_one_out'].flatten().size)],1)
                sseOrth = np.sum((Ycs.flatten() - YtestRaw.flatten()) ** 2)
                errs.append(sseOrth)

            fold_errs[fold-1,:] = np.array(errs)
            sse.append(errs[-1])

        dim_errs.append(np.sum(sse))
    return dim_errs, np.sum(fold_errs,0)


def extract_traj(output_dir, bin_width, data, method='gpfa', x_dim=3, param_cov_type='rbf', param_distance='Euclidean', param_Q = 3, num_folds = 0):
    # num_folds: number of splits (>= 2), set 0 for using all train data

    bin_width = bin_width # in msec # NOT REQUIRED
    min_var_frac = 0.01 # used in em

    # Create results directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Divide data into cross-validation train and test folds
    N = len(data)
    f_div = np.floor(np.linspace(0, N, num_folds+1))

    for cvf in range(num_folds+1):
        if num_folds==0:
            print("\nTraining on all data...\n")
        else:
            print("\nCross-validation fold %d of %d\n" % (cvf, num_folds))
        
        test_mask = np.zeros(N, dtype=bool)
        if cvf > 0:
            test_mask[np.arange(f_div[cvf-1],f_div[cvf], dtype=int)] = True
        train_mask = ~test_mask

        if num_folds == 0:
            # Keep original order if using all data as training set
            tr = np.arange(0, N)
        else:
            tr = np.random.RandomState(seed=42).permutation(N) # deterministic permutation

        train_trial_idx = tr[train_mask]
        test_trial_idx  = tr[test_mask]
        seq_train = [data[trial_num] for trial_num in train_trial_idx] # CHECK if copy.deepcopy() required
        seq_test = [data[trial_num] for trial_num in test_trial_idx]

        print('\nNumber of trials in training: %d\n' % len(seq_train));
        print('\nNumber of trials in testing: %d\n' % len(seq_test));
        print('\nDimensionality of latent space: %d\n' % x_dim);

        if len(seq_train)==0:
            print("No examples in training set. Exiting from current cross-validation run")
            continue

        # If doing cross-validation, don't use private noise variance floor
        # TODO, set minVarFrac and pass to gpfa_engine
        if cvf > 0:
            min_var_frac = -np.inf

        # Name of results file
        output_file = output_dir+"/"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type
        if cvf > 0:
            output_file += "_cv"+str(cvf)
        
        # Call gpfa
        result = None
        result = gpfa_engine(seq_train=seq_train, seq_test=seq_test, fname=output_file,
            x_dim=x_dim, bin_width=bin_width, param_cov_type=param_cov_type, param_distance=param_distance, param_Q = param_Q, min_var_frac=min_var_frac)

        if cvf > 0:
            # Calculate the total sum squared errors for all neurons across all trials in the test sequence
            sse = 0
            for t in range(len(seq_test)):
                q, T = result['seq_test'][t].y.shape
                for m in range(q):
                    C, d, R = result['params'].C, result['params'].d, result['params'].R
                    y = deepcopy(seq_test[t].y).reshape((q*T, 1))
                    y_m = np.delete(seq_test[t].y, m, axis=0).reshape(((q-1)*T, 1)) # all but the mth neuron
                    ym = deepcopy(seq_test[t].y[m]).reshape((T, 1)) # the mth neuron
                    # construct Bm
                    Bm = np.zeros((T, q*T))
                    non_zeros_for_m = np.arange(m*T, (m+1)*T, dtype=int)
                    for j in range(T):
                        Bm[j][non_zeros_for_m[j]] = 1
                    
                    Bm = sparse.csr_matrix(Bm)
                    
                    # construct B_m
                    B_m = np.zeros(((q-1)*T, q*T))
                    non_zeros_without_m = np.zeros((q-1, T), dtype=int)
                    i = 0
                    c = 0
                    for c in range(q):
                        if c != m: # exclude the mth neuron
                            non_zeros_without_m[i] = np.arange(c*T, (c+1)*T)
                            i += 1
                    for i in range(q-1):
                        for j in range(T):
                            B_m[i*T + j][non_zeros_without_m[i][j]] = 1
                    
                    B_m = sparse.csr_matrix(B_m)
                    
                    # construct block matrices for C, d and R
                    C_blah = [C for _ in range(T)]
                    C_block = sparse.block_diag(C_blah)

                    R_blah = [R for _ in range(T)]
                    R_block = sparse.block_diag(R_blah)

                    d_blah = [d.reshape((q, 1)) for _ in range(T)]
                    d_block = np.concatenate(d_blah, 0)
                    
                    K_big, _, _ = make_K_big(result['params'], T)      
                    K_big = sparse.csr_matrix(K_big)

                    # Compute the predicted value of neuron m
                    Sigma = C_block.dot(K_big).dot(C_block.T) + R_block + pow(10, -5) * np.eye(q*T)
                    Sigma = sparse.csr_matrix(Sigma)

                    ym_pred = Bm.dot(d_block) + Bm.dot(Sigma).dot(B_m.T).dot(np.linalg.inv(B_m.dot(Sigma).dot(B_m.T).todense())).dot(y_m - B_m.dot(d_block))

                    sse += np.sum((ym.ravel() - ym_pred.A1)**2) # add prediction error for neuron m
                    
            print(sse)
    # Returns result of the last run fold
    return result