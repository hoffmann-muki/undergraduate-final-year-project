# Classes to store data and parameters for the GPFA model
# Usage:
#
# seq = Seq_Data_Class(trial_id, T, seq_id, y)
# seq.y = np.zeros((10,1))
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np


class Model_Specs:
    def __init__(self, data=None, params=None):
        self.data = data
        self.params = params
        self.bin_width = 200

    def get_data(self):
        return self.data

    # Load data from mat file
    def data_from_mat(self, path_to_mat):
        # By default, data loaded as uint8 which gives errors
        mat_contents = sio.loadmat(path_to_mat, mat_dtype=True)
        data = []
        # Loop over each component of MATLAB struct
        for i in range(len(mat_contents['seq'][0])):
            # Extract trial info
            trial_id = int(mat_contents['seq'][0][i][0][0][0])
            T = int(mat_contents['seq'][0][i][1][0][0])
            seq_id = int(mat_contents['seq'][0][i][2][0][0])
            x = np.zeros(T) # -- mat_contents['seq'][0][i][3] -- when dealing with synthetic data
            y = mat_contents['seq'][0][i][4]
            # Store in data object and append to list
            data.append(Trial_Class(trial_id, T, seq_id, x, y))
        self.data = data

    def populate_sequence(self, s, T, Tmax):
        if len(s) == 0:
            return np.zeros((T,))
        s = s.T[0]
        T2 = int(np.ceil(s[-1]/self.bin_width))
        u = np.zeros((Tmax,)) # for expanding s
        y = np.zeros((T,)) # for storing binned data
        for j in range(len(s)):
            u[int(s[j])-1] = 1 # replace timepoints with spike counts
        for j in range(min(T2, T-1)):
            y[j] = sum(u[j*self.bin_width:(j+1)*self.bin_width])
        if T2 < T-1:
            for j in range(T2, T-1):
                y[j] = sum(u[j*self.bin_width:(j+1)*self.bin_width])
        y[T-1] = sum(u[(T-1)*self.bin_width:])
        return y

    def data_from_movie(self, filepath):
        mat_contents = sio.loadmat(filepath)
        q = mat_contents['n_neurons'][0][0]
        n_trials = mat_contents['n_trials'][0][0]
        y = mat_contents['Spikes']       
        highest_across_trials = np.zeros((n_trials,))
        data = []
        # Y = np.zeros((q,T)) i.e. num of neurons by num of timepoints (per trial)
        for j in range(n_trials):
            for i in range(q):
                elt = y[i][j]
                if len(elt) == 0:
                    continue
                if (max(elt.T[0]) > highest_across_trials[j]):
                    highest_across_trials[j] = max(elt.T[0])
        for j in range(n_trials):
            Ys = []
            Tmax = int(highest_across_trials[j])
            T = int(np.ceil(Tmax/self.bin_width))
            for i in range(q):
                elt = y[i][j]   
                ys = self.populate_sequence(elt, T, Tmax)
                Ys.append(ys)   
            Y = np.asarray(Ys)
            trial_id = (j+1)
            T = Y.shape[1]
            X = np.zeros(T)
            # Store in data object and append to list
            data.append(Trial_Class(trial_id, T, 0, X, Y))            
        self.data = data

    # Helper function to stack variables across trials
    # e.g., sometimes we want vector np.array([data[0].y, data[1].y, data[2].y, ...])
    def stack_attributes(self, attribute):
        if getattr(self.data[0], attribute).size == 1:
            attributes_stacked = np.zeros(len(self.data), dtype='int16')
            for i in range(len(self.data)):
                attributes_stacked[i] = getattr(self.data[i], attribute)
        else:
            attributes_stacked = np.array([]).reshape(
                getattr(self.data[0], attribute).shape[0], 0)
            for i in range(len(self.data)):
                attributes_stacked = np.hstack(
                    (attributes_stacked, getattr(self.data[i], attribute)))
        return attributes_stacked


class Trial_Class:
    def __init__(self, trial_id, T, seq_id, x, y):
        self.trial_id = trial_id
        self.T = T  # TODO: Confusion with transpose
        self.seq_id = seq_id
        self.y = y
        self.x = x
        self.xsm = None
        self.Vsm = None
        self.VsmGP = None

    # Function to print objects
    def __repr__(self):
        return("(Trial id: %d, T: %d, seq id: %d,x: %s, y: %s)"
               % (self.trial_id, self.T, self.seq_id, np.array_repr(self.x), np.array_repr(self.y)))

# This gives the user flexibility to declare parameters, or load them from elsewhere


class Param_Class():
    def __init__(self, param_cov_type=None, param_gamma=None,
                 param_eps=None, param_d=None, param_C=None, param_R=None,
                 param_notes_learnKernelParams=True, param_notes_learnGPNoise=False,
                 param_notes_RforceDiagonal=True, param_q=3, param_p=0, param_lp=0, param_gamma2=0, param_distance='default'):
        self.cov_type = param_cov_type
        self.gamma = param_gamma
        self.distance = param_distance
        self.p = param_p
        self.lp = param_lp
        self.gamma2 = param_gamma2
        self.eps = param_eps
        self.d = param_d
        self.C = param_C
        self.R = param_R
        self.Q = param_q
        self.learnKernelParams = param_notes_learnKernelParams
        self.learnGPNoise = param_notes_learnGPNoise
        self.RforceDiagonal = param_notes_RforceDiagonal

    # Load model parameters from a .mat file
    def params_from_mat(self, path_to_mat):
        mat_contents = sio.loadmat(path_to_mat)
        # Load individual parameters
        content = mat_contents['currentParams'][0][0]
        self.cov_type = content[0][0]
        self.gamma = content[1][0]
        self.eps = content[2][0]
        self.d = content[3].T[0]
        self.C = content[4]
        self.R = content[5]

    # Function to print objects
    def __repr__(self):
        return("Cov type: %s\nGamma: %s\nEps: %s\nd: %s\nC: %s\nR: %s"
               % (self.cov_type, np.array_repr(self.gamma), np.array_repr(self.eps),
                  np.array_repr(self.d), np.array_repr(self.C), np.array_repr(self.R)))
