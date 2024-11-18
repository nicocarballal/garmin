import numpy as np
import scipy
import matplotlib.pyplot as plt

def initialize_mean(t, k):
    #return (np.ones((k,1))*np.array([[t/2, 117]])).reshape((k, 1, 2))
    mus = np.linspace(0, t, k)
    return np.array([np.array([mus[j], 117]) for j in range(k)]).reshape(k, 1, 2)

def initialize_sigma(t,k):
    mus = 117*np.ones((k,))
    ts = np.linspace(t, t, k)
    return np.array([np.diag((ts[j]/k, mus[j])) for j in range(k)])

def initialize_latent(k):
    return 1/k*np.ones((k,))

def gaussian(x, mu, sigma):
  """
  Calculates the Gaussian function for a given x, mean (mu), and standard deviation (sigma).
  """
  return (1 / (np.sqrt(2 * np.pi))) * (1 / np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * ((x - mu) @ np.linalg.inv(sigma) @ (x - mu).T))


def likelihood(n, k, x, mu, sig, pi):
    
    loss = 0
    for k_ in range(k):
        loss += pi[k_] * scipy.stats.norm.pdf(x=x, loc=mu[k_], scale=sig[k_])
    return loss

    
def initialize(t, k):
    mu_arr = initialize_mean(t, k)
    sigma_arr = initialize_sigma(t, k)
    pi_arr = initialize_latent(k)
    return mu_arr, sigma_arr, pi_arr

def plot_multivariate_gaussians(means, covs, colors, t, ax=None):
    """Plots multiple 2D multivariate Gaussian distributions."""

    x, y = np.mgrid[0:t:5, 0:200:1]
    pos = np.dstack((x, y))

    if ax == None:
        fig, ax = plt.subplots()

    for mean, cov, color in zip(means, covs, colors):
        rv = scipy.stats.multivariate_normal(mean, cov)
        ax.contour(x, y, rv.pdf(pos), colors=color)

    plt.show()


def compute(data):
    t, k = 2275, 5
    x = data[:, 0:2].reshape((len(data), 1, 2))
    n = len(data)
    mu_arr, sigma_arr, pi_arr = initialize(t, k)

    S = 10
    mu_val = np.zeros((S + 1, k, 1, 2))
    sigma_val = np.zeros((S + 1, k, 2, 2))
    loss_val = []

    r_nk = np.zeros((n, k))

    mu_val[0, :] = mu_arr
    sigma_val[0] = sigma_arr
    s = 0
    while s < S:
        for n_ in range(n):
            for k_ in range(k):
                num = pi_arr[k_]*gaussian(x[n_], mu_arr[k_], sigma_arr[k_])
                den = np.sum([pi_arr[j]*gaussian(x[n_], mu_arr[j], sigma_arr[j]) for j in range(k)])
                if num == 0:
                    r_nk[n_][k_] = 0
                else:
                    r_nk[n_][k_] = np.divide(num,den)
                

        for k_ in range(k):
            mu_arr[k_] = 1/np.sum(r_nk[:, k_]) * np.sum([r_nk[n_][k_]*x[n_] for n_ in range(n)], axis = 0)
            sigma_arr[k_] = 1/np.sum(r_nk[:, k_])* np.sum([r_nk[n_][k_]*(x[n_]-mu_arr[k_]).T@(x[n_] - mu_arr[k_]) for n_ in range(n)], axis = 0)
            pi_arr[k_] = np.sum(r_nk[:, k_], axis = 0)/n
        
        convergence = likelihood(n, k, x, mu_arr, sigma_arr, pi_arr)
        mu_val[s+1, :] = mu_arr
        sigma_val[s+1, :] = sigma_arr
        loss_val.append(convergence)
        s = s + 1
    
    
        # Plot each Gaussian

    # Define the parameters for multiple Gaussian distributions
    means = mu_val[-1][:, 0]
    covs = sigma_val[-1]

    # Create a figure and axes
    fig, ax = plt.subplots()

    plt.scatter(x[:,0,0], x[:,0,1])
    colors = ['r', 'g', 'b', 'y', 'k']
    plot_multivariate_gaussians(means, covs, colors, t, ax=ax)

    


    


    
    



if __name__ == "__main__":
    data = np.load('data.npy')
    compute(data)