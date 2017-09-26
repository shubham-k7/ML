import os
import os.path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )

args = parser.parse_args()

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X,Y = load_h5py(args.data.strip())

x = np.array(X)
y = np.array(Y)

fig = plt.figure()

plt.scatter(x[:,0],x[:,1],c=y)
plt.title('Plot for Dataset '+ args.data.strip().split('_')[-1][:-3])
fig.savefig('Plots/dataset-'+args.data.strip().split('_')[-1][:-3]+'.png')