import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def VaR_ES_var_covar(x_data, c, w, alpha):
    mu = np.mean(x_data, axis=1)
    Sigma = np.cov(x_data)
    z_alpha = stats.norm.ppf(alpha)
    VaR = -(c + np.dot(w, mu)) + math.sqrt(np.dot(w, np.dot(Sigma, w))) * z_alpha
    ES = -(c + np.dot(w, mu)) + math.sqrt(np.dot(w, np.dot(Sigma, w))) * stats.norm.pdf(z_alpha) / (1 - alpha)
    return VaR, ES


# trading days
td = 252

# level Value at Risk
alpha = 0.98

# import timeseries
n = 5369  # number of trading days
S = np.zeros((5, n))
i = 0
for file in ['BMW.csv', 'Continental.csv', 'SAP.csv',
             'Siemens.csv', 'Volkswagen.csv']:
    with open(file) as csv:
        # change decimal from comma to point
        S[i] = np.flip(np.genfromtxt((line.replace(',', '.') for line in csv), delimiter=';', skip_header=1, usecols=4))
    i += 1

# plot stock prices relative to their starting value to compare performance
plt.plot(range(0, n), S[0, :] / S[0, 0], label='BMW')
plt.plot(range(0, n), S[1, :] / S[1, 0], label='Conti')
plt.plot(range(0, n), S[2, :] / S[2, 0], label='SAP')
plt.plot(range(0, n), S[3, :] / S[3, 0], label='Siem')
plt.plot(range(0, n), S[4, :] / S[4, 0], label='VW')
plt.title('Stock prices')

# initialize matrix for log returns
x = np.zeros((5, n))

for i in range(0, 5):
    # compute logarithmic returns as risk factor changes
    x[i, 1:] = np.diff(np.log(S[i, :]))

# number of shares
alpha_bar = np.array([40, 29, 26, 52, 29])
v = np.dot(S.T, alpha_bar)
# plot portfolio performance
plt.plot(range(0, n), v / v[1], label= 'Portfolio')
plt.legend(loc='upper left', fontsize='x-large')
plt.show()

VaR_var_covar = np.zeros(n)
ES_var_covar = np.zeros(n)

for m in range(td + 1, n - 1):
    c = 0
    w = S[:, m] * alpha_bar
    VaR_var_covar[m + 1], ES_var_covar[m + 1] = VaR_ES_var_covar(x[:, m - td + 1:m], c, w, alpha)

loss = -np.diff(v)
# plot loss, VaR and ES
plt.plot(range(1, n), loss, '+', label = 'Loss')
plt.plot(range(td + 2, n), VaR_var_covar[td + 2:], label = 'VaR')
plt.plot(range(td + 2, n), ES_var_covar[td + 2:], label = 'ES')
plt.legend(loc='upper left', fontsize='x-large')
plt.show()

print('There are ' + str(np.sum(VaR_var_covar[td + 2:] < loss[
                                                         td + 1:])) + ' exceedances of the Losses over the value at risk, where we would expect ' + str(
    (1 - alpha) * (len(loss) - td - 1)) + ' exceedances.')
