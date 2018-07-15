import os
import os.path
import argparse
import h5py
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )

args = parser.parse_args()

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# Preprocess data and split it
X,Y = load_h5py(args.data.strip())

k = 2

x_temp = []

x = np.array(X)
y = np.array(Y)

for xi in x:
	x_temp.append(np.asarray(xi).reshape(-1))

y_temp = []
for yi in y:
	if(yi == 7):
		y_temp.append(0)
	else:
		y_temp.append(1)

x = np.array(x_temp).astype(float)
y = np.array(y_temp)

random_indices=np.random.permutation(len(x))
x1 = []
y1 = []
for i in range(len(random_indices)):
	x1.append(x[random_indices[i]])
	y1.append(y[random_indices[i]])

x = np.array(x1)
y = np.array(y1)

x_sets = np.split(x[:-1],k)
y_sets = np.split(y[:-1],k)

x -= np.mean(x, axis = 0)
max1 = np.max(x) - np.min(x)
x = np.divide(x,max1)

accuracies = 0

iterations = 101
best = -1
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
	best = -1
	for j in range(1,iterations+1,10):
		x_train = np.array(x_train)
		y_train = np.array(y_train)
			
		clf = MLPClassifier(activation="logistic", alpha=0.01, hidden_layer_sizes=(100,50), learning_rate='constant',
		       learning_rate_init=0.01, max_iter=j, momentum=0.95, random_state=1,
		        solver='lbfgs', verbose=False)

		clf.fit(x_train,y_train)
		y_pred = clf.predict(x_test)
		accuracy = np.mean(y_test == y_pred)
		if(accuracy > best):
			best = accuracy
			best_model = clf
		print("Accuracy: ",accuracy)
	accuracies += best

print("average accuracy: ",accuracies/k)