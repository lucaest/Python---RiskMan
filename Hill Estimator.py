import numpy as np
import matplotlib.pyplot as plt

def Hill_Estimator(x, k):
    y = np.flip(np.sort(x))
    alpha = k/ np.sum(np.log(y[0:k]) - np.log(y[k]))
    return alpha


def Hill_Plot(x):
    y = x[x > 0]
    n = len(y)

    a = np.zeros(n-1)
    for k in range(1,n-1):
        a[k-1] = Hill_Estimator(y,k)

    plt.plot(range(1,n), a)


def VaR_ES_Hill(x,p,k):
    n = len (x)
    alpha = Hill_Estimator(x,k)
    y = np.flip(np.sort(x))

    VaR = pow((n/k * (1-p)), -1/alpha) * y[k]
    ES = pow(1-1/alpha, -1) * VaR
    return VaR, ES

n = 500
plt.subplot(3,1,1)
plt.title('Students t distribution with 3 degrees of freedom')
plt.xlabel('k')
plt.ylabel('Hill estimator')
Hill_Plot(np.random.standard_t(3,n))
plt.subplot(3,1,2)
plt.title('Students t distribution with 8 degrees of freedom')
plt.xlabel('k')
plt.ylabel('Hill estimator')
Hill_Plot(np.random.standard_t(8,n))
plt.subplot(3,1,3)
plt.title('Exponential distribution with parameter 1')
plt.xlabel('k')
plt.ylabel('Hill estimator')
Hill_Plot(np.random.exponential(1, n))

#define dataset for the following 

p = 0.98

k1 = 20
VaR, ES = VaR_ES_Hill(dataset,p,k1)
print('k = 20: VaR = ' + str(VaR) + ', ES = ' + str(ES) + ".")

k2 = 50
VaR, ES = VaR_ES_Hill(dataset,p,k2)
print('k = 50: VaR = ' + str(VaR) + ', ES = ' + str(ES) + ".")

VaR = np.zeros(248)
ES = np.zeros(248)

for k in range(1, 249):
    VaR[k-1], ES[k-1] = VaR_ES_Hill(dataset,p,k)

plt.figure()
plt.plot(VaR, label= 'VaR')
plt.plot(ES, label= 'ES')
plt.legend()
plt.title('VaR and ES of given data using hill estimation as a function of k')
plt.show()
# %%
