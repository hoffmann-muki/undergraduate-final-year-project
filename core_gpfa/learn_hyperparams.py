
import numpy as np
from core_gpfa.util import invToeplitz
from scipy.optimize import minimize
from core_gpfa.make_K_big import nn_kernel, Lee_metric, Tsum, Discrete_metric, circular_kernel, logit_kernel


def learn_GP_params(seq, current_params):
    # Learn GP hyperparameters for each latent dimension separately
    # This code uses the BFGS optimizer which estimates the gradient

    isVerbose = False

    oldParams = current_params.gamma.copy()
    xDim = len(oldParams)
    out_params = []

    if current_params.cov_type == 'rbf':
        fname = grad_rbf
    elif current_params.cov_type == 'sm':
        fname = grad_sm
    elif current_params.cov_type == 'rq':
        fname = grad_rq
    elif current_params.cov_type == 'pw':
        fname = grad_pw
    elif current_params.cov_type == 'im':
        fname = grad_im
    elif current_params.cov_type == 'p':
        fname = grad_p
    elif current_params.cov_type == 'lp':
        fname = grad_lp
    elif current_params.cov_type == 'cos':
        fname = grad_cos
    elif current_params.cov_type == 'lin':
        fname = grad_lin
    elif current_params.cov_type == 'poly':
        fname = grad_poly
    elif current_params.cov_type == 'nn':
        fname = grad_nn
    elif current_params.cov_type == 'circ':
        fname = grad_circ
    elif current_params.cov_type == 'logit':
        fname = grad_logit

    precomp = makePrecomp(seq, xDim, distance=current_params.distance)

    for i in range(xDim):

        initp = np.log(oldParams[i])           # Single param per latent dim
        curr_args = {
            'Tall': precomp['Tall'][i],
            'T': precomp['T'][i],
            'Tdif': precomp['Tdif'],
            'difSq': precomp['difSq'][i],
            'numTrials': precomp['numTrials'][i],
            'PautoSUM': precomp['PautoSUM'][i],
            'distance': current_params.distance
        }
        if current_params.cov_type != 'sm':

            const = current_params.eps[i]
            res = minimize(fun=fname,
                           x0=initp,
                           args=(curr_args, const),
                           method='BFGS',
                           options={'disp': isVerbose})
            out_params.append(np.exp(res.x[0]))

        elif current_params.cov_type == 'sm':
            Q = current_params.Q
            # Weights must sum to 1
            wbound = tuple((-10, None) for _ in range(Q))
            gaussbound = tuple((None, None) for _ in range(Q*2))
            bnds = wbound + gaussbound
            res = minimize(fun=fname,
                           x0=initp,
                           args=(curr_args, Q),
                           method='L-BFGS-B',
                           bounds=bnds,
                           options={'disp': isVerbose}),
            res = np.exp(res[0].x)
            res[:Q] = res[:Q] / np.sum(res[:Q])
            out_params.append(res.tolist())

    return out_params


def makePrecomp(seq, xDim, distance):

    Tall = np.array([trial.T for trial in seq])
    Tmax = np.max(Tall)
    Tdif = np.tile(np.arange(1, Tmax+1, 1).reshape(Tmax, 1), (1, Tmax)) - np.tile(np.arange(1, Tmax+1, 1), (Tmax, 1))

    absDif = []
    difSq = []
    Talll = []

    for i in range(xDim):
        absDif.append(abs(Tdif))
        difSq.append(np.square(Tdif))
        Talll.append(Tall)

    # This is assumed to be unique - Tu is a scalar
    # the code won't work if Tu is a vector
    Tu = np.unique(Talll)
    nList = []
    T = []
    numTrials = []
    PautoSUM = []

    #  Loop once for each state dimension (each GP)
    for i in range(xDim):
        nList.append(np.where(Tall == Tu)[0])
        T.append(Tu)
        numTrials.append(nList[i].size)
        PautoSUM.append(np.zeros((Tu[0], Tu[0])))

    # Loop once for each dimension
    for i in range(xDim):
        # Loop once for each trial in dimension
        for j in nList[i]:
            PautoSUM[i] = PautoSUM[i] + seq[j].VsmGP[:, :, i] + \
                np.outer(seq[j].xsm[i, :].T, seq[j].xsm[i, :])

    precomp = {
        'T': T,
        'Tall': Talll,
        'Tdif': Tdif,
        'difSq': difSq,
        'numTrials': numTrials,
        'PautoSUM': PautoSUM,
        'distance': distance}

    return precomp


def grad_rbf(p, curr_args, const):
    # Cost function for squared exponential kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    if curr_args['distance'] == 'Root Manhattan':
        temp = (1-const) * (np.exp(-np.exp(p) / 2 * np.abs(curr_args['Tdif'])) + 0.5 * np.eye(Tmax))
    elif curr_args['distance'] == 'Lee':
        temp = (1-const) * np.exp(-0.5 * Lee_metric(Tmax)**2)
    elif curr_args['distance'] == 'Canberra':
        temp = (1-const) * np.exp(-0.5 * (np.abs(curr_args['Tdif'])/Tsum(Tmax))**2)
    elif curr_args['distance'] == 'Discrete':
        temp = (1-const) * np.exp(-0.5 * Discrete_metric(Tmax)**2)
    else:
        temp = (1-const) * np.exp(-np.exp(p) / 2 * curr_args['difSq'])
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_rq(p, curr_args, const):
    # Cost function for rational quadratic kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    if curr_args['distance'] == 'Root Manhattan':
        temp = (1-const) * ((1 - (np.exp(p) * np.abs(curr_args['Tdif']))/(np.exp(p) * np.abs(curr_args['Tdif']) + 1)) + np.eye(Tmax))
    elif curr_args['distance'] == 'Lee':
        temp = (1-const) * (1 - (Lee_metric(Tmax)**2)/(Lee_metric(Tmax)**2 + 1))
    elif curr_args['distance'] == 'Canberra':
        temp = (1-const) * (1 - ((np.abs(curr_args['Tdif'])/Tsum(Tmax))**2)/((np.abs(curr_args['Tdif'])/Tsum(Tmax))**2 + 1))
    elif curr_args['distance'] == 'Discrete':
        temp = (1-const) * (1 - (Discrete_metric(Tmax)**2)/(Discrete_metric(Tmax)**2 + 1))
    else:
        temp = (1-const) * (1 - (np.exp(p) * curr_args['difSq'])/(np.exp(p) * curr_args['difSq'] + 1))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_pw(p, curr_args, const):
    # Cost function for Paley-Weiner kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    if curr_args['distance'] == 'Root Manhattan':
        temp = (1-const) * (np.sinc(0.5 * np.exp(p) * np.sqrt(np.abs(curr_args['Tdif']))) + np.eye(Tmax))
    elif curr_args['distance'] == 'Lee':
        temp = (1-const) * (np.sinc(0.5 * np.exp(p) * Lee_metric(Tmax)) + 5 * np.eye(Tmax))
    elif curr_args['distance'] == 'Canberra':
        temp = (1-const) * (np.sinc(2 * (np.abs(curr_args['Tdif']))/Tsum(Tmax)) + 0.1 * np.eye(Tmax))
    elif curr_args['distance'] == 'Discrete':
        temp = (1-const) * np.sinc(2 * Discrete_metric(Tmax)**2)
    else:
        temp = (1-const) * np.sinc(0.5 * np.exp(p) * np.abs(curr_args['Tdif']))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_im(p, curr_args, const):
    # Cost function for inverse multiquadric kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    if curr_args['distance'] == 'Root Manhattan':
        temp = (1-const) * (1/np.sqrt(np.exp(p) * np.abs(curr_args['difSq']) + 1))
    elif curr_args['distance'] == 'Lee':
        temp = (1-const) * (1/np.sqrt(Lee_metric(Tmax)**2 + 1))
    elif curr_args['distance'] == 'Canberra':
        temp = (1-const) * (1/np.sqrt((np.abs(curr_args['Tdif'])/Tsum(Tmax))**2 + 1))
    elif curr_args['distance'] == 'Discrete':
        temp = (1-const) * (1/np.sqrt(Discrete_metric(Tmax)**2 + 1))
    else:
        temp = (1-const) * (1/np.sqrt(np.exp(p) * curr_args['difSq'] + 1))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_p(p, curr_args, const):
    # Cost function for periodic kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    if curr_args['distance'] == 'Lee':
        temp = (1-const) * (np.exp(np.cos(np.exp(p) * Lee_metric(Tmax))) + 5 * np.eye(Tmax))
    elif curr_args['distance'] == 'Discrete':
        temp = (1-const) * (np.exp(np.cos(np.exp(p) * Discrete_metric(Tmax))) + 5 * np.eye(Tmax))
    else:    
        temp = (1-const) * (np.exp(np.cos(np.exp(p) * np.abs(curr_args['Tdif']))) + 5 * np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_lp(p, curr_args, const):
    # Cost function for locally periodic kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) \
    * (np.exp(np.cos(np.exp(p) * np.abs(curr_args['Tdif']))) + 5 * np.eye(Tmax)) \
    * np.exp(-0.5 * curr_args['difSq'])
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_cos(p, curr_args, const):
    # Cost function for cosine kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) * (np.cos(np.exp(p) * np.abs(curr_args['Tdif'])) + np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_lin(p, curr_args, const):
    # Cost function for linear kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) \
    * (np.array([[0 if i==j else i*j/(Tmax**2) for j in range(1,Tmax+1)] for i in range(1,Tmax+1)]) \
    + np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f

def grad_poly(p, curr_args, const):
    # Cost function for polynomial kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) \
    * (np.array([[0 if i==j else (i*j)**2/(Tmax**4) for j in range(1,Tmax+1)] for i in range(1,Tmax+1)]) \
    + np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_nn(p, curr_args, const):
    # Cost function for neural network kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) * np.array([[nn_kernel(i,j) for j in range(1,Tmax+1)] for i in range(1,Tmax+1)])
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_circ(p, curr_args, const):
    # Cost function for circular kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) * (circular_kernel(Tmax) + np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_logit(p, curr_args, const):
    # Cost function for circular kernel

    Tall = curr_args['Tall']
    Tmax = np.max(Tall)
    temp = (1-const) * (logit_kernel(Tmax) + np.eye(Tmax))
    Kmax = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f


def grad_sm(p, curr_args, Q):
    # Cost function for spectral mixture kernel
    # No gradient is returned

    p = np.exp(p).tolist()
    # Force weights to sum to 1
    # w = (p[:Q] / np.sum(p[:Q])).tolist()
    w = p[:Q]
    m = p[Q:Q*2]
    v = p[Q*2:Q*3]
    # Generate the covariance for given setting of parameters
    Kmax = np.zeros(curr_args['difSq'].shape)
    for i in range(Q):
        Kmax = Kmax + w[i] * np.exp(-2 * np.pi**2 * v[i]**2 * curr_args['difSq']) * \
            np.cos(2 * np.pi * curr_args['Tdif'].T * m[i])
    Kmax = Kmax + 0.00001*np.identity(Kmax.shape[0])

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 *
          np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f
