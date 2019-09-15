import numpy as np 

numPts = 100
mtrx = np.random.rand(numPts, 3)
mtrx[:, 2] = np.arange(numPts)
np.savetxt('test.txt', mtrx, delimiter = '\t', header = 'A\tB\tC', comments = '')
