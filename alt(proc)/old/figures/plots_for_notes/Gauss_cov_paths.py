import numpy as np 
import matplotlib.pyplot as plt

def Gaussiancov(x, y, lam = 1.0):
	return np.exp(-1.0/(2.0*lam**2) * np.linalg.norm(x-y)**2)
#	return np.exp(-np.sqrt(5)*np.linalg.norm(x-y))*(1 + np.sqrt(5) *np.linalg.norm(x-y) + 5/3 * np.linalg.norm(x-y)**2)
y = np.linspace(0,1,100)
m = np.zeros(len(y))
K = np.zeros((len(y), len(y)))
for i in range(len(y)):
	for j in range(len(y)):
		K[i,j] = Gaussiancov(y[i], y[j])

Z1 = np.zeros((len(y), 2))
Z2 = np.zeros((len(y), 2))
Z3 = np.zeros((len(y), 2))
Z4 = np.zeros((len(y), 2))
Z5 = np.zeros((len(y), 2))
Z1[:,0] = y
Z2[:,0] = y
Z3[:,0] = y
Z4[:,0] = y
Z5[:,0] = y

np.random.seed(3)
Z1[:,1] = np.random.multivariate_normal(m,K)
Z2[:,1] = np.random.multivariate_normal(m,K)
Z3[:,1] = np.random.multivariate_normal(m,K)
Z4[:,1] = np.random.multivariate_normal(m,K)
Z5[:,1] = np.random.multivariate_normal(m,K)

np.savetxt('Gausscov1.txt', Z1)
np.savetxt('Gausscov2.txt', Z2)
np.savetxt('Gausscov3.txt', Z3)
np.savetxt('Gausscov4.txt', Z4)
np.savetxt('Gausscov5.txt', Z5)
