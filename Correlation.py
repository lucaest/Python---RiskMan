import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def Kendall(x):
    n = len(x[:,0])
    tau = np.zeros(n-1)

    for k in range(0, n-1):
        tau[k] = np.sum(np.sign((x[k,0]-x[k+1:,0]) * (x[k,1] - x[k+1:,1]).T))

    # nchoosek returns the binomial coefficient n over 2 = n(n-1)/2
    tau = np.sum(tau) * 1/special.binom(n,2)
    return tau

# compute ranks of a vector
def vrank(x):
    n = len(x)
    idx = np.argsort(x)
    r = np.zeros(n)
    for k in range(0,n):
        r[idx[k]] = k
    return r

def Spearman(x):
    n = len(x[:,0])

    #compute ranks for both components
    r_x1 = vrank(x[:,0])
    r_x2 = vrank(x[:,1])

    #calculate Spearman's rho
    rho = 12/n/(n**2 -1) * np.dot((r_x1- 0.5*(n+1)),(r_x2 - 0.5*(n+1)))
    return rho


# import timeseries
n = 5369  # number of trading days
S = np.zeros((2, n))
i = 0
for file in ['Continental.csv','Volkswagen.csv']:
    with open(file) as csv:
        # change decimal from comma to point
        S[i] = np.flip(np.genfromtxt((line.replace(',', '.') for line in csv), delimiter=';', skip_header=1, usecols=4))
    i += 1

# initialize matrix for log returns
x = np.zeros((n, 2))

for i in range(0, 2):
    # compute logarithmic returns as risk factor changes
    x[1:,i] = np.diff(np.log(S[i, :]))

# estimate mean vector and covariance matrix
mu = np.mean(x, axis=0)
Sigma = np.cov(x, rowvar=False)

# estimate linear correlation
rho = Sigma[0,1]/np.sqrt(Sigma[0,0]) /np.sqrt(Sigma[1,1])

print('time series: linear correlation = ' + str(rho) + ', Kendalls tau = ' + str(Kendall(x)) + ', Spearmans rho = ' + str(Spearman(x)) + '.')

# plot common log returns
z = np.zeros(n)
plt.subplot(2,1,1)
plt.plot(x[:,0], x[:,1], 'o')
plt.plot(x[:,1], z, 'r')
plt.plot(z, x[:,1], 'r')
plt.xlabel('Continental')
plt.ylabel('Volkswagen')
plt.title('Common log returns of Continental and Volkswagen stocks')

# part d)
# simulate random variables
y = np.random.multivariate_normal(mu, Sigma, n)
# plot
plt.subplot(2,1,2)
plt.plot(y[:,0], y[:,1], 'o')
plt.plot(y[:,0], z, 'r')
plt.plot(z, y[:,1], 'r')
plt.title('Simulation')
plt.xlabel('X_1')
plt.ylabel('X_2')

#estimate correlations from simulated data
print('Simulation: Kendalls tau = ' + str(Kendall(y)) + ', Spearmans rho = ' + str(Spearman(y)) + '.')
plt.show()