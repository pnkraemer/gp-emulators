from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as sts
def Gaussiancov(x, y, lam = 1.0):
	return np.exp(-1.0/(2.0*lam**2) * np.linalg.norm(x-y)**2)
def materncov(x, y, lam = 1.0):
	return np.exp(-np.sqrt(5.0)*np.linalg.norm(x-y))*(1.0 + np.sqrt(5.0)*np.linalg.norm(x-y) + 5.0/3.0 * np.linalg.norm(x-y)**2)

def truesol(x):
	return 0.5* np.sin(3*3.14159265*x)


#materncov = Gaussiancov
np.random.seed(1)


#	return np.exp(-np.sqrt(5)*np.linalg.norm(x-y))*(1 + np.sqrt(5) *np.linalg.norm(x-y) + 5/3 * np.linalg.norm(x-y)**2)
y = np.linspace(0,1,2)
m = np.zeros(len(y))
K = np.zeros((len(y), len(y)))
for i in range(len(y)):
	for j in range(len(y)):
		K[i,j] = materncov(y[i], y[j])



Kinv = np.linalg.inv(K)
rhs = truesol(y) # + np.random.normal(0,0.1,len(y))

evalpts = np.linspace(0,1,50)

K2 = np.zeros((len(evalpts), len(y)))
for i in range(len(evalpts)):
	for j in range(len(y)):
		K2[i,j] = materncov(evalpts[i], y[j])

K3 = np.zeros((len(evalpts), len(evalpts)))
for i in range(len(evalpts)):
	for j in range(len(evalpts)):
		K3[i,j] = materncov(evalpts[i], evalpts[j])


mnew = K2.dot(Kinv.dot(rhs))
newcov = K3 - K2.dot(Kinv.dot(K2.T))

KK = np.diag(newcov)

am = np.argmax(KK)





np.random.seed(2)
Z1 = np.zeros((len(evalpts), 2))
Z1[:,0] = evalpts

length = 200
minimum = np.zeros(length)






plt.figure()
plt.subplot(331)
for i in range(length):
	Z1[:,1] = np.random.multivariate_normal(mnew,newcov)
	minimum[i] = evalpts[np.argmin(Z1[:,1])]
	plt.plot(Z1[:,0], Z1[:,1], color = "gray")
plt.plot(evalpts, mnew, linewidth = 2, color = "black")
plt.plot(y, rhs, 'o', markersize = 7, color = "darkslategray")
plt.plot(evalpts[am], mnew[am], '^', markersize = 10, color = "red")
plt.xlim((0,1.0))
plt.ylim((-1.0, 1.0))




# augment
numRds = 8
for idx in range(numRds):
	print "augmenting...", idx + 1
	ynew = np.zeros(len(y) + 1)
	ynew[0:len(y)] = y
	ynew[len(y)] = evalpts[am]
	y = ynew

	rhsnew = np.zeros(len(rhs) + 1)
	rhsnew[0:len(rhs)] = rhs
	rhsnew[len(rhs)] = truesol(evalpts[am])#  + np.random.normal(0,0.1,1)
	rhs = rhsnew

	m = np.zeros(len(y))
	K = np.zeros((len(y), len(y)))
	for i in range(len(y)):
		for j in range(len(y)):
			K[i,j] = materncov(y[i], y[j])



	Kinv = np.linalg.inv(K)
	#rhs = truesol(y)  + np.random.normal(0,0.1,len(y))

	evalpts = np.linspace(0,1,200)

	K2 = np.zeros((len(evalpts), len(y)))
	for i in range(len(evalpts)):
		for j in range(len(y)):
			K2[i,j] = materncov(evalpts[i], y[j])

	K3 = np.zeros((len(evalpts), len(evalpts)))
	for i in range(len(evalpts)):
		for j in range(len(evalpts)):
			K3[i,j] = materncov(evalpts[i], evalpts[j])


	mnew = K2.dot(Kinv.dot(rhs))
	newcov = K3 - K2.dot(Kinv.dot(K2.T))

	KK = np.diag(newcov)

	am = np.argmax(KK)





	Z1 = np.zeros((len(evalpts), 2))
	Z1[:,0] = evalpts

	length = 20
	minimum = np.zeros(length)





	n = (332 + idx)
	plt.subplot(n)
	for i in range(length):
		Z1[:,1] = np.random.multivariate_normal(mnew,newcov)
		minimum[i] = evalpts[np.argmin(Z1[:,1])]
		plt.plot(Z1[:,0], Z1[:,1], color = "gray")
	plt.plot(evalpts, mnew, linewidth = 2, color = "black")
	plt.plot(y, rhs, 'o', markersize = 7, color = "darkslategray")
	plt.xlim((0,1.0))
	plt.ylim((-1.0, 1.0))
	plt.plot(evalpts[am], mnew[am], '^', markersize = 10, color = "red")





plt.show()


# np.savetxt('exponcov1.txt', Z1)
# np.savetxt('exponcov2.txt', Z2)
# np.savetxt('exponcov3.txt', Z3)
# np.savetxt('exponcov4.txt', Z4)
# np.savetxt('exponcov5.txt', Z5)
