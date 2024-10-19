import numpy as np
from scipy.stats import invgamma
import scipy
import matplotlib.pyplot as plt

def initialize_mean(t, k):
    return 117*np.ones((k,))

def initialize_sigma(t,k):
    return t/k*np.ones((k,))

def initialize_latent(k):
    return 1/k*np.ones((k,))

def initialize_gamma(k):
    a_arr = .01 * np.ones((k,))
    b_arr = .01 * np.ones((k,))
    return a_arr, b_arr

def gaussian(x, mu, sigma):
  """
  Calculates the Gaussian function for a given x, mean (mu), and standard deviation (sigma).
  """
  return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sum_of_gaussians(k, x_arr, mu_arr, sigma_arr):
    return np.sum([gaussian(x_arr[i], mu_arr[i], sigma_arr[i]) for i in range(k)])

def data_moments(data,k):
    return np.mean(data)*np.ones((k,)), np.std(data)**2*np.ones((k,))

def inv_gamma(a, b):
    # Create an Inverse Gamma distribution object
    inverse_gamma_dist = invgamma(a, scale=b)

    return inverse_gamma_dist.rvs(size=1)

def likelihood(n, k, x, mu, sig, pi):
    
    loss = 0
    for k_ in range(k):
        loss += pi[k_] * scipy.stats.norm.pdf(x=x, loc=mu[k_], scale=sig[k_])
    return loss

    
def initialize(t, k):
    mu_arr = initialize_mean(t, k)
    sigma2_arr = initialize_sigma(t, k)
    a_arr, b_arr = initialize_gamma(k)
    pi_arr = initialize_latent(k)
    return mu_arr, sigma2_arr, pi_arr

def compute(data):
    t, k = 50, 3
    x = data[:, 1]
    n = len(data)
    mu_0, sigma_0 = data_moments(x, k)
    mu_arr, sigma_arr, pi_arr = initialize(t, k)


    S = 100
    mu_val = np.zeros((S + 1, k))
    sigma_val = np.zeros((S + 1, k))
    loss_val = []

    r_nk = np.zeros((n, k))

    mu_val[0, :] = mu_arr
    sigma_val[0,:] = sigma_arr
    s = 0
    while s < S:
        for n_ in range(n):
            for k_ in range(k):
                num = pi_arr[k_]*gaussian(x[n_], mu_arr[k_], sigma_arr[k_])
                den = np.sum([pi_arr[j]*gaussian(x[n_], mu_arr[j], sigma_arr[j]) for j in range(k)])
                r_nk[n_][k_] = np.divide(num,den)

        for k_ in range(k):
            mu_arr[k_] = 1/np.sum(r_nk[:, k_]) * np.sum([r_nk[n_][k_]*x[n_] for n_ in range(n)])
            sigma_arr[k_] = 1/np.sum(r_nk[:, k_])* np.sum([r_nk[n_][k_]*(x[n_]-mu_arr[k_])**2 for n_ in range(n)])
            pi_arr[k_] = np.sum(r_nk[:, k_])/n
        
        convergence = likelihood(n, k, x, mu_arr, sigma_arr, pi_arr)
        mu_val[s+1, :] = mu_arr
        sigma_val[s+1, :] = sigma_arr
        loss_val.append(convergence)
        s = s + 1
    
    plt.figure()
    plt.hist(x, density=True)
        # Plot each Gaussian
    for i in range(k):
        plt.plot(x, gaussian(x, mu_val[-1][i], sigma_val[-1][i]))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multiple Gaussians')
    plt.show()
    print('hey')



    


    
    



if __name__ == "__main__":
    data = np.load('data.npy')
    compute(data)