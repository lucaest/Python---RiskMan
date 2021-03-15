import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def qqplot(x, F_inv):
    # length of data
    n = len(x)

    # sort data
    x_sorted = np.flip(np.sort(x))

    # compute theoretical quantiles of reference distribution
    y = np.linspace(n/(n+1), 1/(n+1), num= n)
    y = F_inv(y)

    # plot sorted data against quantiles of reference distribution
    plt.plot(x_sorted, y, 'ob')

    # linear regression. Alternative with np.poly1d and np.polyfit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_sorted, y)
    plt.plot(x_sorted, intercept+slope*x_sorted, '-r')

    #label plots
    plt.xlabel('Data: empirical quantiles')
    plt.ylabel('Theoretical quantiles')


# load DAX time series
dax = np.flip(np.genfromtxt('dax_data.csv', delimiter=';', skip_header=1, usecols=4))
# compute log returns
x = np.diff(np.log(dax))

#define reference distributions
#standard normal
def Phi_inv(x):
    return stats.norm.ppf(x)
#t distribution with 2 degrees of freedom
def T_inv_2(x):
    return stats.t.ppf(x, df = 2)
# t distribution with 5 degrees of freedom
def T_inv_5(x):
    return stats.t.ppf(x, df = 5)
#t distribution with 8 degrees of freedom
def T_inv_8(x):
    return stats.t.ppf(x, df=8)

#plots
# import timeseries
n = 5369  # number of trading days
S = np.zeros(n)
i = 1
for file in ['BMW.csv']:
    with open(file) as csv:
        # change decimal from comma to point
        data = np.flip(np.genfromtxt((line.replace(',', '.') for line in csv), delimiter=';', skip_header=1, usecols=4))

    x = np.diff(np.log(data))
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(4,5,i)
    qqplot(x, Phi_inv)
    plt.title('N(0,1)')
    plt.subplot(4,5,5+i)
    qqplot(x, T_inv_2)
    plt.title('t- distribution with 2 df')
    plt.subplot(4,5, 10+i)
    qqplot(x, T_inv_5)
    plt.title('t- distribution with 5 df')
    plt.subplot(4, 5, 15 + i)
    qqplot(x, T_inv_8)
    plt.title('t- distribution with 8 df')
    i += 1
plt.tight_layout()
plt.show()
# %%
