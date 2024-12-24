import numpy as np
from scipy.linalg import pinv

def sq_dist(X1, X2=None):
    """Calculate squared distance matrix between two sets of points."""
    if X2 is None:
        X2 = X1
    return np.add.outer(np.sum(X1**2, axis=1), np.sum(X2**2, axis=1)) - 2*np.dot(X1, X2.T)

def matern(distmat, tau2, theta, g, v):
    """Matern covariance function."""
    n1, n2 = distmat.shape
    covmat = np.zeros((n1, n2))
    
    if v == 0.5:
        r = np.sqrt(distmat/theta)
        covmat = tau2 * np.exp(-r)
    elif v == 1.5:
        r = np.sqrt(3*distmat/theta)
        covmat = tau2 * (1 + r) * np.exp(-r)
    elif v == 2.5:
        r = np.sqrt(5*distmat/theta)
        covmat = tau2 * (1 + r + r**2/3) * np.exp(-r)
    
    if n1 == n2:
        np.fill_diagonal(covmat, covmat.diagonal() + tau2 * g)
    
    return covmat

def exp2(distmat, tau2, theta, g):
    """Exponential covariance function."""
    n1, n2 = distmat.shape
    covmat = np.zeros((n1, n2))
    
    r = distmat/theta
    covmat = tau2 * np.exp(-r)
    
    if n1 == n2:
        np.fill_diagonal(covmat, covmat.diagonal() + tau2 * g)
    
    return covmat

def krig(y, dx, d_new=None, d_cross=None, theta=None, g=None, tau2=1, s2=False, 
         sigma=False, v=2.5, prior_mean=0, prior_mean_new=0):
    """Kriging prediction."""
    if v is None:
        C = exp2(dx, 1, theta, g)
        C_cross = exp2(d_cross, 1, theta, 0) if d_cross is not None else None
    else:
        C = matern(dx, 1, theta, g, v)
        C_cross = matern(d_cross, 1, theta, 0, v) if d_cross is not None else None
    
    C_inv = pinv(C)
    mean = prior_mean_new + C_cross @ C_inv @ (y - prior_mean)
    
    output = {'mean': mean}
    
    if s2:
        C_new = (1 + g) * np.ones((d_new.shape[0], 1))
        output['s2'] = tau2 * (C_new - np.diag(C_cross @ C_inv @ C_cross.T))
    
    if sigma:
        quadterm = C_cross @ C_inv @ C_cross.T
        if v is None:
            C_new = exp2(d_new, 1, theta, g)
        else:
            C_new = matern(d_new, 1, theta, g, v)
        output['sigma'] = tau2 * (C_new - quadterm)
    
    return output