import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import optimize

# negative log-likelihood
def logL_GPD(theta_hat, y):
    Nu = len(y)
    A = np.log(1 + theta_hat[0] / theta_hat[1] * y)
    y = -1 * (-Nu * np.log(theta_hat[1]) - (1 / theta_hat[0] + 1) * np.sum(A))
    return y


def PoT_estimated(x, u):
    theta_0 = np.array([0.1, 1])
    y = x[x > u]
    y = y - u
    # optimize the negative ll in theta with given boundarys
    theta_hat = optimize.minimize(logL_GPD, args=y, x0=theta_0, bounds=((0.0001, 1), (0.0001, 100)))
    beta_hat = theta_hat.x[1]
    gamma_hat = theta_hat.x[0]
    return beta_hat, gamma_hat


def VaR_ES_PoT(x, p, u):
    beta_hat, gamma_hat = PoT_estimated(x, u)
    n = len(x)
    y = x[x > u]
    y = y - u
    Nu = len(y)
    VaR = u + beta_hat / gamma_hat * ((n / Nu * (1 - p)) ** -gamma_hat - 1)
    ES = VaR + (beta_hat + gamma_hat * (VaR - u)) / (1 - gamma_hat)
    return VaR, ES

#...MEP
print("For the threshold u the values 6.6 or 8.9 seem to make sense")
u1 = 6.6
VaR, ES = VaR_ES_PoT(dataset, p, u1)
print("The estimates for u = 6.6 the VaR and ES using the POT method are " + str(VaR) + " for the 98% VaR and " + str(
    ES) + " for the ES.")
u1 = 8.9
VaR, ES = VaR_ES_PoT(dataset, p, u1)
print("The estimates for u = 8.9 the VaR and ES using the POT method are " + str(VaR) + " for the 98% VaR and " + str(
    ES) + " for the ES.")
plt.show()

# %%
