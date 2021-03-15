# C-Exercise 09
# Luca Estivariz
# Lorenz Klose


#%%
#packages
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import datetime


# --------part a)
def VaR_ES_var_covar (x_data, c, w, alpha):
    #empirical distribution parameters
    mu_hat = x_data.mean(axis=0)
    sigma_hat = np.cov(np.transpose(x_data))
    #quantile
    q = norm.ppf(alpha, 0, 1)
    #linearized loss operator
    l_x_delta = - (c + np.dot(w, mu_hat))
    #convenience
    factor = np.sqrt(np.dot(np.dot(np.transpose(w), sigma_hat), w))
    #estimated VaR and ES
    VaR_hat = l_x_delta + factor * q
    Es_hat = l_x_delta + (factor * norm.pdf(q)) / (1-alpha)

    return VaR_hat, Es_hat

# --------part b)
#import data
Bmw = pd.read_csv('BMW.csv', delimiter=';', decimal=',')
Sap = pd.read_csv('SAP.csv', delimiter=';', decimal=',')
Vw = pd.read_csv('VW.csv', delimiter=';', decimal=',')
Continental = pd.read_csv('Continental.csv', delimiter=';', decimal=',')
Siemens = pd.read_csv('Siemens.csv', delimiter=';', decimal=',')
stocklist = (Bmw, Sap, Vw, Continental, Siemens)
#sort by date
for i in stocklist:
    i.sort_values(by=['Datum'], inplace=True, ascending=True)
    i['Datum'] = [datetime.datetime.strptime(j, '%Y-%m-%d') for j in i['Datum']]
    i['Datum'] = [datetime.datetime.date(j) for j in i['Datum']] 
#save date 
time = np.asarray(Bmw['Datum'])
#compute risk factor matrix
x_data = np.zeros((len(time)-1, 5))
x_data = np.transpose([np.diff(np.log(i['Schlusskurs'])) for i in stocklist])

# --------part c)
alpha = 0.98
alpha_bar = np.asarray([40, 29, 26, 52, 29])
n = 252
#m=254
S = np.asarray([Bmw['Schlusskurs'], Sap['Schlusskurs'], Vw['Schlusskurs'], Continental['Schlusskurs'], Siemens['Schlusskurs']])
#portfolio value
V = np.dot(alpha_bar, S)
#allocate memory
VaR_hat = np.zeros(len(time))
Es_hat = np.zeros(len(time))
#calculate var and es 
#only for m>254
for m in range(n+1, len(time)-1):
    c = 0
    w = np.multiply(alpha_bar, np.transpose(S[:, m]))#weights for each asset
    VaR_hat[m+1] = VaR_ES_var_covar(x_data[(m-n+1):m, :], c, w, alpha)[0]
    Es_hat[m+1] = VaR_ES_var_covar(x_data[(m-n+1):m, :], c, w, alpha)[1]
#get loss vector
loss = np.zeros(len(time))
loss[1:] = -np.diff(V)
#plot
plt.clf()
plt.plot(time, loss, ',', color='g', label='Portfolio Loss')
plt.plot(time, VaR_hat, color='b', label='VaR_hat', linewidth=0.5)
plt.plot(time, Es_hat, color='r', label='Es_hat', linewidth=0.5)
plt.title('Variance-Covariance VaR and ES')
plt.ylim(0)#for better visibility
plt.legend()
plt.show()

# %%
