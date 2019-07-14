from joblib import Parallel, delayed
import multiprocessing
import time
import numpy as np
# what are your inputs, and what operation do you want to 
# perform on each input. For example...

def do_seq(inp):
	time.sleep(0.1 * np.random.rand())
	return inp * np.ones(3)








def do(inp, ncores):
	def processInput(i):
		time.sleep(0.1 * np.random.rand())
		return i * np.ones(3)

	Ret = Parallel(n_jobs=ncores)(delayed(processInput)(i) for i in inp)
	return Ret

inputs = range(250)
num_cores = multiprocessing.cpu_count()

print("number of cores:", num_cores)
retu = do(inputs, num_cores)
print("ayt")
for i in inputs:
	do_seq(i)