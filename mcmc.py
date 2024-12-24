import numpy as np
from scipy.stats import gamma
from scipy.linalg import pinv, det
from tooling import sq_dist, exp2, matern

def logl(out_vec, in_dmat, g, theta, outer=True, v=None, tau2=False, mu=0, scale=1):
    """Calculate log-likelihood."""
    n = len(out_vec)
    if v is None:
        K = scale * exp2(in_dmat, 1, theta, g)
    else:
        K = scale * matern(in_dmat, 1, theta, g, v)
    
    K_inv = pinv(K)
    diff = out_vec - mu
    quadterm = diff.T @ K_inv @ diff
    
    detK = det(K)
    if detK == 0:
        detK = np.finfo(float).eps 
    
    if outer:
        logl_val = (-n*0.5)*np.log(quadterm) - 0.5*np.log(abs(detK))
    else:
        logl_val = -0.5*np.log(abs(detK)) - 0.5*quadterm
    
    tau2_val = quadterm/n if tau2 else None 
    
    return {'logl': logl_val, 'tau2': tau2_val}

def sample_g(out_vec, in_dmat, g_t, theta, alpha, beta, l, u, ll_prev=None, v=None):
    """Sample g parameter."""
    g_star = l*g_t/u + (u*g_t/l - l*g_t/u)*np.random.rand()
    
    ru = np.random.rand()
    if ll_prev is None:
        temp = logl(out_vec, in_dmat, g_t, theta, True, v)
        ll_prev = temp['logl']
    
    lpost_threshold = (ll_prev + np.log(gamma.pdf(g_t - np.sqrt(np.finfo(float).eps), 
                      alpha, scale=1/beta)) + np.log(ru) - np.log(g_t) + np.log(g_star))
    
    temp = logl(out_vec, in_dmat, g_star, theta, True, v)
    ll_new = temp['logl']
    
    new = ll_new + np.log(gamma.pdf(g_star - np.sqrt(np.finfo(float).eps), alpha, scale=1/beta))
    
    if new > lpost_threshold:
        g = g_star
        ll = ll_new
    else:
        g = g_t
        ll = ll_prev
    
    return {'g': g, 'll': ll}

def sample_theta(out_vec, in_dmat, g, theta_t, alpha, beta, l, u, outer=True, 
                ll_prev=None, v=None, tau2=False, prior_mean=0, scale=1):
    """Sample theta parameter."""
    theta_star = l*theta_t/u + (u*theta_t/l - l*theta_t/u)*np.random.rand()
    
    ru = np.random.rand()
    if ll_prev is None:
        temp = logl(out_vec, in_dmat, g, theta_t, outer, v, False, prior_mean, scale)
        ll_prev = temp['logl']
    
    lpost_threshold = (ll_prev + np.log(gamma.pdf(theta_t - np.sqrt(np.finfo(float).eps), 
                      alpha, scale=1/beta)) + np.log(ru) - np.log(theta_t) + np.log(theta_star))
    
    ll_new = logl(out_vec, in_dmat, g, theta_star, outer, v, tau2, prior_mean, scale)
    
    new = ll_new['logl'] + np.log(gamma.pdf(theta_star - np.sqrt(np.finfo(float).eps), 
                                           alpha, scale=1/beta))
    
    if new > lpost_threshold:
        theta = theta_star
        ll = ll_new['logl']
        tau2_val = ll_new['tau2']
    else:
        theta = theta_t
        ll = ll_prev
        tau2_val = None
    
    return {'theta': theta, 'll': ll, 'tau2': tau2_val}

def sample_w(out_vec, wt, wt_dmat, in_dmat, g, theta_y, theta_w, ll_prev=None, 
             v=None, prior_mean=None, scale=1):
    """Sample w parameter."""
    D = wt.shape[1]
    
    if prior_mean is None:
        prior_mean = np.zeros_like(wt)
    
    if ll_prev is None:
        temp = logl(out_vec, wt_dmat, g, theta_y, True, v)
        ll_prev = temp['logl']
    
    for i in range(D):
        if v is None:
            cov = scale * exp2(in_dmat, 1, theta_w[i], 0)
        else:
            cov = scale * matern(in_dmat, 1, theta_w[i], 0, v)
        
        w_prior = np.random.multivariate_normal(prior_mean[:, i], cov)
        
        a = 2*np.pi*np.random.rand()
        amin = a - 2*np.pi
        amax = a
        
        ru = np.random.rand()
        ll_threshold = ll_prev + np.log(ru)
        
        accept = False
        count = 0
        w_prev = wt[:, i].copy()
        
        while not accept:
            count += 1
            
            wt[:, i] = w_prev * np.cos(a) + w_prior * np.sin(a)
            dw = sq_dist(wt)
            
            temp = logl(out_vec, dw, g, theta_y, True, v)
            new_logl = temp['logl']
            
            if new_logl > ll_threshold:
                ll_prev = new_logl
                accept = True
            else:
                if a < 0:
                    amin = a
                else:
                    amax = a
                a = amin + (amax-amin)*np.random.rand()
                if count > 100:
                    raise RuntimeError('reached maximum iteration of ESS')
    
    return {'w': wt, 'll': ll_prev, 'dw': dw}

def gibbs_sampling(x, y, nmcmc, D, verb=True, initial=None, true_g=None, settings=None, v=None):
    """Perform Gibbs sampling for DGP."""
    if initial is None:
        initial = {
            'w': np.tile(x, [1, D]),
            'theta_y': 0.1,
            'theta_w': 0.1,
            'g': 0.01,
            'tau2': 1
        }
    
    dx = sq_dist(x)
    dw = sq_dist(initial['w'])
    
    g = np.full(nmcmc, np.nan)
    g[0] = true_g if true_g is not None else initial['g']
    
    theta_y = np.full(nmcmc, np.nan)
    theta_y[0] = initial['theta_y']
    
    theta_w = np.full((nmcmc, D), np.nan)
    theta_w[0, :] = initial['theta_w']
    
    w = [None] * nmcmc
    w[0] = initial['w'].copy()
    
    tau2 = np.full(nmcmc, np.nan)
    tau2[0] = initial['tau2']
    
    ll_store = np.full(nmcmc, np.nan)
    ll_outer = None
    
    for j in range(1, nmcmc):
        if verb and (j+1) % 500 == 0:
            print(j+1)
        
        if true_g is None:
            samp = sample_g(y, dw, g[j-1], theta_y[j-1], settings['alpha']['g'], 
                          settings['beta']['g'], settings['l'], settings['u'], ll_outer, v)
            g[j] = samp['g']
            ll_outer = samp['ll']
        else:
            g[j] = true_g
        
        samp = sample_theta(y, dw, g[j], theta_y[j-1], settings['alpha']['theta_y'], 
                          settings['beta']['theta_y'], settings['l'], settings['u'], 
                          True, ll_outer, v, True)
        theta_y[j] = samp['theta']
        ll_outer = samp['ll']
        tau2[j] = tau2[j-1] if samp['tau2'] is None else samp['tau2'].squeeze()
        
        for i in range(D):
            prior_mean = np.zeros(x.shape[0])
            w_temp = w[j-1].copy()
            samp = sample_theta(w_temp[:, i], dx, np.sqrt(np.finfo(float).eps), 
                              theta_w[j-1, i], settings['alpha']['theta_w'], 
                              settings['beta']['theta_w'], settings['l'], settings['u'], 
                              False, None, v, False, prior_mean, settings['inner_tau2'])
            theta_w[j, i] = samp['theta']
        
        prior_mean = np.zeros((x.shape[0], D))
        samp = sample_w(y, w[j-1].copy(), dw, dx, g[j], theta_y[j], theta_w[j, :], 
                       ll_outer, v, prior_mean, settings['inner_tau2'])
        w[j] = samp['w']
        ll_outer = samp['ll']
        ll_store[j] = ll_outer.squeeze()
        dw = samp['dw']
    
    return {
        'g': g,
        'theta_y': theta_y,
        'theta_w': theta_w,
        'w': w,
        'tau2': tau2,
        'll': ll_store
    }