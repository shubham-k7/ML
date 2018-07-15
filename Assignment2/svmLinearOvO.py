import os
import os.path
import argparse
import h5py
import numpy as np
import csv
from sklearn import svm
import operator
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )

args = parser.parse_args()

def my_predict(mesh,x_train,y_train,c_opt):
	row = []
	for m in M:
		y_train_norm = []
		for j in range(len(y_train)):
			if(m != y_train[j]):
				y_train_norm.append(-1)
			else:
				y_train_norm.append(1)

		clf = svm.SVC(kernel = 'linear',C=c_opt)
		clf.fit(x_train,y_train_norm)

		sv = clf.support_vectors_
		dc = clf.dual_coef_
		d = clf.intercept_
		
		y_temp_test = []
		for j in range(len(mesh)):
			sum1 = 0
			for o in range(len(dc[0])):
				sum1 = sum1 + dc[0][o]*np.dot(mesh[j],sv[o])
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

M = np.unique(y)
M.sort()

C = [0.1,1,10]
accuracies = []
for c in C:
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

		row = np.zeros((len(x_test),len(M)))
		# print(row)
		for m1i in range(len(M)-1):
			for m2i in range(m1i+1,len(M)):
				y_trainM1vM2 = []
				x_trainM1vM2 = []
				for j in range(len(y_train)):
					if(M[m1i] == y_train[j]):
						x_trainM1vM2.append(x_train[j])
						y_trainM1vM2.append(1)
					elif(M[m2i]==y_train[j]):
						x_trainM1vM2.append(x_train[j])
						y_trainM1vM2.append(-1)

				# print(x_trainM1vM2)
				# print(y_trainM1vM2)
				clf = svm.SVC(kernel = 'linear',C=c)
				clf.fit(x_trainM1vM2,y_trainM1vM2)

				sv = clf.support_vectors_
				dc = clf.dual_coef_
				d = clf.intercept_
				
				for j in range(len(x_test)):
					sum1 = 0
					for o in range(len(dc[0])):
						sum1 = sum1 + dc[0][o]*np.dot(x_test[j],sv[o])
					sum1 = sum1+d[0]
					
					# if sum1 > 0: jisse 1 assign kara tha, usse +1 
					if(sum1 >= 0):
						row[j][M[m1i]] += 1
					# if sum1 < 0: jisse -1 assign kara tha, usse +1
					else:
						row[j][M[m2i]] += 1
		
		y_pred = []
		for itemp in row:
			index, value = max(enumerate(itemp), key=operator.itemgetter(1))
			y_pred.append(index)
		accuracy = accuracy + np.mean(y_pred == y_test)

		for j in range(len(y_pred)):
			confusion_matrix[y_pred[j]][y_test[j]] += 1
	
	print(confusion_matrix)
	accuracy = accuracy/k
	print(accuracy)
	accuracies.append(accuracy)

index, value = max(enumerate(accuracies), key=operator.itemgetter(1))
print("Optimum C",C[index],"Accuracy",value)
c_optimum = C[index]

x_train = x
y_train = y

fig = plt.figure()
h = 0.06

min_x, max_x = x[:, 0].min() - 1, x[:, 0].max() + 1
min_y, max_y = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y, h))
Z = my_predict(np.c_[xx.ravel(), yy.ravel()],x,y,c_optimum)
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.axis('tight')
plt.title('Plot for Decision Boundary '+ args.data.strip().split('_')[-1][:-3])
# fig.savefig('Plots/svmLinearOvO-decision-'+args.data.strip().split('_')[-1][:-3]+'.png')


for i in range(len(sv)):
	plt.plot(sv[i][0],sv[i][1],'g+')
fig.savefig('Plots/svmLinearOVO-DB-SV-'+args.data.strip().split('_')[-1][:-3]+'.png')

for i in range(len(sv)):
	plt.plot(sv[i][0],sv[i][1],'*')
fig.savefig('Plots/svmLinearOvO-SV-'+args.data.strip().split('_')[-1][:-3]+'.png')
