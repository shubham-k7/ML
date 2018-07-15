import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()

# Assumed that the data is .h5 format
f = h5py.File('./'+args.data.strip(),'r')

x = f['X']
y = f['Y']

colors = []
for i in y:
	for j in range(10):
		if(i[j]==1):
			colors.append(j)

fig = plt.figure()

x_embedded = TSNE(n_components = 2,random_state = 0).fit_transform(x)

plt.scatter(x_embedded[:,0],x_embedded[:,1],c=colors)

# Tha data is stored in Directory of choice with name graph.png
fig.savefig(args.plots_save_dir.strip()+args.data.strip().split('/')[-1][:-3]+'.png')