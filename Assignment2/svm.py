import os
import os.path
import argparse
import h5py
import numpy as np
import csv
from sklearn import svm
import operator

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )

args = parser.parse_args()

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X,Y = load_h5py(args.data.strip())

random_indices=np.random.permutation(len(X))
x = []
y = []
for i in range(len(random_indices)):
	x.append(X[random_indices[i]])
	y.append(Y[random_indices[i]])

x = np.array(x)
y = np.array(y)

k = 5

x_sets = np.split(x,k)
y_sets = np.split(y,k)

M = []

for i in range(len(y)):
	if(y[i] not in M):
		M.append(y[i])

table = []

accuracy = 0
for i in range(k):
	x_train = []
	y_train = []
	for j in range(len(x_sets)):
		if(i != j):
			x_train.extend(x_sets[j])
			y_train.extend(y_sets[j])
		else:
			x_test = x_sets[i]
			y_test = y_sets[i]

	row = []
	for m in sorted(M):
		y_trainMvA = []
		for j in range(len(y_train)):
			if(m != y_train[j]):
				y_trainMvA.append(-1)
			else:
				y_trainMvA.append(1)

		clf = svm.SVC(kernel = 'linear')
		clf.fit(x_train,y_trainMvA)

		sv = clf.support_vectors_
		dc = clf.dual_coef_
		d = clf.intercept_
		
		y_temp_test = []
		for j in range(len(x_test)):
			sum1 = 0
			for o in range(len(dc[0])):
				sum1 = sum1 + dc[0][o]*np.dot(x_test[j],sv[o])
			y_temp_test.append(sum1+d[0])
		
		row.append(y_temp_test)

	y_pred = []
	for itemp in np.transpose(np.array(row)):
		index, value = max(enumerate(itemp), key=operator.itemgetter(1))
		y_pred.append(index)
	
	accuracy = accuracy + np.mean(y_pred == y_test)

accuracy = accuracy/k
print(accuracy)