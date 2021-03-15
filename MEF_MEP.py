import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import optimize


def MEF(x, u):
    if u >= np.max(x):
        print("MEF: u has to be smaller than the largest value of x.")
        sys.exit()
    y = x > u
    N_u = np.sum(y)
    e = 1 / N_u * np.dot((x - u), y)
    return e


def MEP(x):
    n = len(x)
    x = np.flip(np.sort(x))

    e = np.zeros(n - 1)
    for k in range(1, n):
        e[k - 1] = MEF(x, x[k])

    plt.plot(x[1:], e, "o")


n = 500
plt.subplot(3, 1, 1)
plt.title('Students t distribution with 3 degrees of freedom')
plt.xlabel('Threshold u')
plt.ylabel('Empirical mean excess function')
MEP(np.random.standard_t(3, n))
plt.subplot(3, 1, 2)
plt.title('Students t distribution with 8 degrees of freedom')
plt.xlabel('Threshold u')
plt.ylabel('Empirical mean excess function')
MEP(np.random.standard_t(8, n))
plt.subplot(3, 1, 3)
plt.title('Exponential distribution with parameter 1')
plt.xlabel('Threshold u')
plt.ylabel('Empirical mean excess function')
MEP(np.random.exponential(1, n))
