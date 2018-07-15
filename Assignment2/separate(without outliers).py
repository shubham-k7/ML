import os
import os.path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn import svm

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

h = 0.07

def my_kernel(X,Y):
	K = np.zeros((X.shape[0],Y.shape[0]))
	for i in range(len(X)):
		for j in range(len(Y)):
			K[i,j] = np.exp(-1*np.linalg.norm(X[i]-Y[j])**2)
	return K

classifier = svm.SVC(kernel=my_kernel)
classifier.fit(x,y)

fig = plt.figure()

# min_x, max_x = x[:, 0].min() - 1, x[:, 0].max() + 1
# min_y, max_y = x[:, 1].min() - 1, x[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y, h))
# Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Put the result into a color plot

# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
# plt.axis('tight')
# plt.title('Plot for Dataset '+ args.data.strip().split('_')[-1][:-3])
# fig.savefig('Plots/separated-'+args.data.strip().split('_')[-1][:-3]+'.png')

y_test = classifier.predict(x)
x_temp = []
y_temp = []
for i in range(len(y_test)):
	if(y_test[i]==y[i]):
		x_temp.append(x[i])
		y_temp.append(y[i])

# classifier.fit(x_temp,y_temp)
x_temp = np.array(x_temp)
y_temp = np.array(y_temp)

min_x, max_x = x_temp[:, 0].min() - 1, x_temp[:, 0].max() + 1
min_y, max_y = x_temp[:, 1].min() - 1, x_temp[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x_temp[:, 0], x_temp[:, 1], c=y_temp, cmap=plt.cm.Paired, edgecolors='k')
plt.axis('tight')
plt.title('Plot for Dataset '+ args.data.strip().split('_')[-1][:-3])
# plt.show()
fig.savefig('Plots/separated(removed outliers)-'+args.data.strip().split('_')[-1][:-3]+'.png')

