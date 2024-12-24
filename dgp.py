import numpy as np
from mcmc import gibbs_sampling
from tooling import sq_dist, krig

def fit_dgp(x, y, nmcmc=10000, D=None, verb=True, w_0=None, g_0=0.01, 
            theta_y_0=0.1, theta_w_0=0.1, true_g=None, cov="matern", v=2.5):
    """Fit Deep Gaussian Process."""
    if D is None:
        D = x.shape[1]
    
    if cov == "exp2":
        v = None
    
    if w_0 is None:
        w_0 = np.tile(x, [1, D])
    
    settings = {
        'l': 1,
        'u': 2,
        'alpha': {'g': 1.5, 'theta_w': 1.5, 'theta_y': 1.5},
        'beta': {'g': 3.9, 'theta_w': 3.9/4, 'theta_y': 3.9/6},
        'inner_tau2': 1
    }
    
    initial = {
        'w': w_0,
        'theta_y': theta_y_0,
        'theta_w': theta_w_0,
        'g': g_0,
        'tau2': 1
    }
    
    if cov == "matern" and v not in [0.5, 1.5, 2.5]:
        raise ValueError('v must be one of 0.5, 1.5 or 2.5')
    
    output = {
        'x': x,
        'y': y,
        'nmcmc': nmcmc,
        'settings': settings,
        'v': v
    }
    
    samples = gibbs_sampling(x, y, nmcmc, D, verb, initial, true_g, settings, v)
    output.update(samples)
    
    return output

def predict_dgp(object, x_new):
    """Make predictions using fitted DGP model."""
    object['x_new'] = x_new
    n_new = x_new.shape[0]
    D = object['w'][0].shape[1]
    dx = sq_dist(object['x'])
    d_cross = sq_dist(x_new, object['x'])
    mu_t = np.full((n_new, object['nmcmc']), np.nan)
    sigma_sum = np.zeros((n_new, n_new))
    
    for t in range(object['nmcmc']):
        w_t = object['w'][t]
        w_new = np.full((n_new, D), np.nan)
        
        for i in range(D):
            k = krig(w_t[:, i], dx, None, d_cross, object['theta_w'][t, i], 
                    np.sqrt(np.finfo(float).eps), object['settings']['inner_tau2'], 
                    None, None, object['v'])
            w_new[:, i] = k['mean']
        
        k = krig(object['y'], sq_dist(w_t), sq_dist(w_new), sq_dist(w_new, w_t), 
                object['theta_y'][t], object['g'][t], object['tau2'][t], 
                False, True, object['v'])
        mu_t[:, t] = k['mean'][:,0]
        sigma_sum += k['sigma']
    
    mu_cov = np.cov(mu_t)
    object['mean'] = np.mean(mu_t, axis=1)
    object['sigma'] = sigma_sum/object['nmcmc'] + mu_cov
    
    return object