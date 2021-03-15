import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def VaR_ES_log_normal(s, alpha):
    X = np.diff(np.log(s))
    mu = np.mean(X)
    sigma = np.sqrt(np.var(X))
    quantile = stats.norm.ppf(alpha)
    VaR = s[-1] * (1 - np.exp(mu - sigma * quantile))
    ES = s[-1] * (1 - 1 / (1 - alpha) * np.exp(mu + 0.5 * sigma ** 2) * stats.norm.cdf(-quantile - sigma))
    return VaR, ES


def VaR_ES_historic(x_data, l, alpha):
    l_data_sorted = np.flip(np.sort(l(x_data)))
    n = len(l_data_sorted)
    VaR = l_data_sorted[int(np.floor(n * (1 - alpha)) + 1)]
    ES = 1 / (np.floor(n * (1 - alpha)) + 1) * np.sum(l_data_sorted[0:int(np.floor(n * (1 - alpha)) + 1)])
    return VaR, ES


# trading days
td = 252

# level Value at Risk
alpha = np.array([0.9, 0.95])

# import time series
dax = np.flip(np.genfromtxt('dax_data.csv', delimiter=';', skip_header=1, usecols=4))

x = np.diff(np.log(dax))

n = len(dax)
m = len(alpha)

VaR_historic = np.zeros((n, m))
ES_historic = np.zeros((n, m))
VaR_lognormal = np.zeros((n, m))
ES_lognormal = np.zeros((n, m))

# compute VaR and ES with historical simulation and log normal method
for k in range(td + 1, n - 1):

    # define loss operator and compute VaR and ES with historical simulation
    def l(x):
        return dax[k] * (1 - np.exp(x))


    for i in range(0, m):
        VaR_historic[k + 1, i], ES_historic[k + 1, i] = VaR_ES_historic(x[k - td + 1:k], l, alpha[i])

        # compute VaR and ES with log normal method
        VaR_lognormal[k + 1, i], ES_lognormal[k + 1, i] = VaR_ES_log_normal(dax[k - td:k], alpha[i])

loss = -np.diff(dax)
for i in range(0, m):
    # plot VaR and ES
    plt.figure()
    plt.plot(loss, '+', label = 'Loss')
    plt.plot(range(td + 2, n), VaR_historic[td + 2:, i], label = 'VaR historic')
    plt.plot(range(td + 2, n), ES_historic[td + 2:, i], label = 'ES historic')
    plt.title('Historical simulation with alpha = ' + str(alpha[i] * 100) + "%")
    plt.legend(loc='upper left', fontsize='x-large')

    # compare lognormal method and historical simulation in a new plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(VaR_historic[td + 2:, i], label = 'VaR historic')
    plt.plot(VaR_lognormal[td + 2:, i], label = 'VaR lognormal')
    plt.title('Value at Risk with alpha = ' + str(alpha[i] * 100) + '%')
    plt.legend(loc='upper left', fontsize='x-large')

    # Expected Shortfall
    plt.subplot(2, 1, 2)
    plt.plot(ES_historic[td + 2:, i], label = 'ES historic')
    plt.plot(ES_lognormal[td + 2:, i], label = 'ES lognormal')
    plt.title('Expected Shortfall with alpha = ' + str(alpha[i] * 100) + '%')
    plt.legend(loc='upper left', fontsize='x-large')

viol_hist = np.zeros((n,m))
viol_lognormal = np.zeros((n,m))

for i in range(0, m):
    viol_hist[td+2:, i] = loss[td + 1:] > VaR_historic[td + 2:, i]
    viol_lognormal[td+2:, i] = loss[td + 1:] > VaR_lognormal[td + 2:, i]

print('For alpha = ' + str(
    alpha[0] * 100) + '%: ' + 'Using the method of historic simulation to estimate the VaR we observe ' + str(np.sum(
    viol_hist[:, 0]))+' violations, while using the method of iid normally distributed log-returns we observe ' + str(
    np.sum(viol_lognormal[:, 0]))+' violations. The expected number of violations is 10% x ' + str(
    len(loss)) + ' = ' + str(len(loss) * 0.1) + '.')

print('For alpha = ' + str(
    alpha[1] * 100) + '%: ' + 'Using the method of historic simulation to estimate the VaR we observe ' + str(np.sum(
    viol_hist[:, 1]))+' violations, while using the method of iid normally distributed log-returns we observe ' + str(
    np.sum(viol_lognormal[:, 1]))+' violations. The expected number of violations is 5% x ' + str(
    len(loss)) + ' = ' + str(len(loss) * 0.05) + '.')

plt.show()
# Explanation: There is a lag in detecting phases of high volatility by historic simulations, hence the underestimation
# of the VaR in times of high volatility, which leads to too much violations.
