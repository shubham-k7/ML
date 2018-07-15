import os
import os.path
import argparse
import h5py
import numpy as np
import csv
from sklearn import svm
from sklearn.manifold import TSNE
import operator
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )

args = parser.parse_args()


def my_predict(mesh,x_train,y_train,gamma_optimum):
	row = []
	for m in M:
		y_train_norm = []
		for j in range(len(y_train)):
			if(m != y_train[j]):
				y_train_norm.append(-1)
			else:
				y_train_norm.append(1)

		clf = svm.SVC(kernel='rbf',gamma=gamma_optimum)
		clf.fit(x_train,y_train_norm)

		sv = clf.support_vectors_
		dc = clf.dual_coef_
		d = clf.intercept_
		
		y_temp_test = []
		for j in range(len(mesh)):
			sum1 = 0
			for o in range(len(dc[0])):
				sum1 = sum1 + dc[0][o]*np.exp(-1*np.linalg.norm(mesh[j]-sv[o])**2)
			y_temp_test.append(sum1+d[0])

		row.append(y_temp_test)

	# print(row)
	y_pred = []
	for itemp in np.transpose(np.array(row)):
		# print(itemp)
		index, value = max(enumerate(itemp), key=operator.itemgetter(1))
		y_pred.append(index)

	# print(y_pred)
	return np.array(y_pred)

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X,Y = load_h5py(args.data.strip())
X = TSNE(n_components = 2,random_state = 0).fit_transform(X)
newY = []
for i in Y:
	for j in range(len(i)):
		if(i[j]==1):
			newY.append(j)

Y = newY

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

M = np.unique(y)
M.sort()

G = [0.1,0.7,0.9]

accuracies = []
for g in G:
	accuracy = 0
	confusion_matrix = np.zeros((len(M),len(M)))
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
		for m in M:
			y_trainMvA = []
			for j in range(len(y_train)):
				if(m != y_train[j]):
					y_trainMvA.append(-1)
				else:
					y_trainMvA.append(1)

			clf = svm.SVC(kernel='rbf',gamma=g)
			clf.fit(x_train,y_trainMvA)

			sv = clf.support_vectors_
			dc = clf.dual_coef_
			d = clf.intercept_
			
			y_temp_test = []
			for j in range(len(x_test)):
				sum1 = 0
				for o in range(len(dc[0])):
					sum1 = sum1 + dc[0][o]*np.exp(-1*np.linalg.norm(x_test[j]-sv[o])**2)
				y_temp_test.append(sum1+d[0])
			
			row.append(y_temp_test)

		y_pred = []
		for itemp in np.transpose(np.array(row)):
			index, value = max(enumerate(itemp), key=operator.itemgetter(1))
			y_pred.append(index)
		
		accuracy = accuracy + np.mean(y_pred == y_test)
		if(g==0.7):
			for j in range(len(y_pred)):
				confusion_matrix[y_pred[j]][y_test[j]] += 1
			# print("confusion_matrix for k: "+str(i))
			
	print(confusion_matrix)
	accuracy = accuracy/k
	print(accuracy)
	accuracies.append(accuracy)

index, value = max(enumerate(accuracies), key=operator.itemgetter(1))
print("Optimum Gamma",G[index],"Accuracy",value)
g_optimum = G[index]

x_train = x
y_train = y

fig = plt.figure()
h = 0.06

min_x, max_x = x[:, 0].min() - 1, x[:, 0].max() + 1
min_y, max_y = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y, h))
Z = my_predict(np.c_[xx.ravel(), yy.ravel()],x,y,g_optimum)
Z = Z.reshape(xx.shape)

# Put the result into a color plot

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.axis('tight')
plt.title('Plot for Decision Boundary '+ args.data.strip().split('_')[-1][:-3])
# fig.savefig('Plots/svmRbfOvR-decision-'+args.data.strip().split('_')[-1][:-3]+'.png')

for i in range(len(sv)):
	plt.plot(sv[i][0],sv[i][1],'g+')
fig.savefig('Plots/svmRbfOVR-DB-SV-'+args.data.strip().split('_')[-1][:-3]+'.png')