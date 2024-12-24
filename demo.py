import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dgp import fit_dgp, predict_dgp

# Demo script
def run_demo():
    """Run demonstration of Deep Gaussian Process."""
    
    # Define test function
    def f(x):
        return np.sin(2*np.pi*2*(1+2*x**2)*x)
    
    # Generate training data
    n = 50
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = f(x)
    
    # Generate test data
    np_test = 200
    xp = np.linspace(0, 1, np_test).reshape(-1, 1)
    yp = f(xp)
    
    # Plot test function
    plt.figure(figsize=(10, 6))
    plt.plot(xp, yp)
    plt.scatter(x, y, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Function')
    plt.grid(True)
    plt.show()
    
    # Deep Gaussian Process prediction
    np.random.seed(123)
    nmcmc = 1000
    
    # Fit DGP
    results = fit_dgp(x, y, nmcmc)
    
    # Make predictions
    results = predict_dgp(results, xp)
    
    # Plot MCMC diagnostics
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].plot(results['ll'])
    axs[0, 0].set_title('Log-likelihood')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(results['g'])
    axs[0, 1].set_title('g')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(results['theta_y'])
    axs[1, 0].set_title('theta_y')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(results['theta_w'])
    axs[1, 1].set_title('theta_w')
    axs[1, 1].grid(True)
    
    # Plot warping function samples
    gs = axs[0, 2].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    step = max(1, round(nmcmc/1000))
    for idx in range(0, nmcmc, step):
        axbig.plot(results['x'], results['w'][idx])
    axbig.set_xlabel('x')
    axbig.set_ylabel('w')
    axbig.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(results['x'], results['y'], c='b', marker='o', label='Training Data')
    plt.plot(xp, yp, 'g.', label='True Function')
    plt.plot(results['x_new'], results['mean'], 'r-', label='DGP Mean')
    
    # Calculate confidence intervals
    lb = results['mean'] + norm.ppf(0.05, 0, np.sqrt(np.diag(results['sigma'])))
    ub = results['mean'] + norm.ppf(0.95, 0, np.sqrt(np.diag(results['sigma'])))
    
    plt.plot(results['x_new'], lb, 'm--', label='90% Confidence Interval')
    plt.plot(results['x_new'], ub, 'm--')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('DGP with Two Layers')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_demo()