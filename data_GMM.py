import numpy as np
from scipy.stats import invgamma
import matplotlib.pyplot as plt

def initialize_mean(t, k):
    return np.linspace(0, t, k)

def initialize_sigma(t,k):
    return .01 * t/k*np.ones((k,))**2

def initialize_gamma(k):
    a_arr = .01 * np.ones((k,))
    b_arr = .01 * np.ones((k,))
    return a_arr, b_arr

def gaussian(x, mu, sigma2):
    return -1 / (-np.sqrt(sigma2)*np.sqrt(2*np.pi)) * np.exp(-0.5 * (((x - mu)/np.sqrt(sigma2))**2))

def sum_of_gaussians(k, x_arr, mu_arr, sigma_arr):
    return np.sum([gaussian(x_arr[i], mu_arr[i], sigma_arr[i]) for i in range(k)])

def data_moments(data,k):
    return np.mean(data)*np.ones((k,)), np.std(data)**2*np.ones((k,))

def inv_gamma(a, b):
    # Create an Inverse Gamma distribution object
    inverse_gamma_dist = invgamma(a, scale=b)

    return inverse_gamma_dist.rvs(size=1)

def initialize(t, k):
    mu_arr = initialize_mean(t, k)
    sigma2_arr = initialize_sigma(t, k)
    a_arr, b_arr = initialize_gamma(k)
    return mu_arr, sigma2_arr, a_arr, b_arr




def compute(data):
    t, k = 50, 3
    x = data[:, 1]
    n = len(data)
    mu_0, sigma2_0 = data_moments(x, k)
    mu_arr, sigma2_arr, a_arr, b_arr = initialize(t, k)


    S = 100
    mu_val = np.zeros((S + 1, k))
    sig2_val = np.zeros((S + 1, k))
    mu_n_val = np.zeros((S + 1, k))


    mu_n = (np.divide(mu_0,sigma2_0) + np.divide(n*mu_0,sigma2_arr))/(1/sigma2_0 + n/sigma2_arr)* np.ones((k, ))  
    mu_val[0, :] = mu_arr
    mu_n_val[0, :] = mu_n
    
    for s in range(100):
        mu_n = (np.divide(mu_0,sigma2_0) + np.divide(n*mu_0,sigma2_arr))/(1/sigma2_0 + n/sigma2_arr)* np.ones((k, ))
        sigma2_n = 1 / (1/sigma2_0 + n/sigma2_0)
        mu_arr = [np.random.normal(mu_n[i], sigma2_n[i], 1) for i in range(k)]
        mu_arr = np.array([mu[0] for mu in mu_arr])
        sigma2_arr = [inv_gamma(a_arr[i] + n/2, b_arr[i] + 1/2*np.sum((data[:,1] - mu_arr[i])**2)) for i in range(k)]
        sigma2_arr = np.array([sigma2[0] for sigma2 in sigma2_arr])

        mu_val[s + 1, :] = mu_arr
        mu_n_val[s + 1, :] = mu_n
        sig2_val[s + 1, :] = sigma2_arr
    
    plt.figure()
    plt.hist(x, density=True)
        # Plot each Gaussian
    for i in range(k):
        plt.plot(x, gaussian(x, mu_val[-1][i], sig2_val[-1][i]))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multiple Gaussians')
    plt.show()
    print('hey')



    


    
    



if __name__ == "__main__":
    data = np.load('data.npy')
    compute(data)