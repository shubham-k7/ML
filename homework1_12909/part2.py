import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# Preprocess data and split it
X,Y = load_h5py(args.train_data.strip())

y = np.array(Y)

y_temp = []
for i in y:
	for j in range(len(i)):
		if(i[j]==1):
			y_temp.append(j)

x_train = np.array(X[:int(len(X)*0.7)])
y_train = np.array(y_temp[:int(len(X)*0.7)])
x_test = np.array(X[int(len(X)*0.7):])
y_test = np.array(y_temp[int(len(Y)*0.7):])
y_temp = np.array(y_temp)
# k cross validation k
k = 5
x_training = []
y_training = []
x_testing = []
y_testing = []
x_sets = np.split(X,k)
y_sets = np.split(y_temp,k)

for i in range(k):
	trainX = []
	trainY =[]
	for j in range(len(x_sets)):
		if(i != j):
			trainX += x_sets
			trainY += y_sets
		else:
			y_testing.append(y_sets[i])
			x_testing.append(x_sets[i])
	
	x_training.extend(trainX)
	y_training.extend(trainY)

	# x_training.append([x_sets[index] for index in range(len(x_sets)) if ((index+1) != i) ])

print y_training
# ,y_testing

# Train the models

if args.model_name == 'GaussianNB':
	nb = GaussianNB()
	nb.fit(x_train,y_train)
	y_pred = nb.predict(x_test)
	print y_pred
	print y_test
	accuracy = np.mean(y_test == y_pred)
	print str(accuracy*100)+'%'

elif args.model_name == 'LogisticRegression':
	penalty = ['l1','l2']
	C = [0.5,1.0,1.5]
	max_iter = [100,150,200]
	parameters = []
	accuracies = []
	for p in penalty:
		for c in C:
			for m in max_iter:
				for i in range(k):
					lr = LogisticRegression(penalty=p , C=c , max_iter=m)
					lr.fit(x_training[i],y_training[i])				
					y_pred = lr.predict(x_testing[i])
					accuracy = np.mean(y_testing[i] == y_pred)
					# print str(accuracy*100)+'%'
					parameters.append([p,c,m])
					accuracies.append(accuracy*100)
	print parameters
	print accuracies
	# x_labels=[]
	# for x in parameters:
	# 	# x_labels.append('pen='+str(x[0])+',C='+str(x[1])+',max_iter='+str(x[2]))
	# 	x_labels.append(str(x[0])+','+str(x[1])+','+str(x[2]))
	
	fig = plt.figure()
	plt.bar(np.arange(len(accuracies)),accuracies)
	plt.xticks(np.arange(len(accuracies)), [str(x) for x in np.arange(len(accuracies))])
	# plt.show()

	fig.savefig(args.plots_save_dir.strip() + args.train_data.strip().split('/')[-1][:-3] +'_'+args.model_name +'.png')

elif args.model_name == 'DecisionTreeClassifier':
	max_depth=[5,7,10]
	min_samples_split=[2,3,4]
	min_samples_leaf=[1,2,3]
	parameters = []
	accuracies = []
	for p in max_depth:
		for c in min_samples_split:
			for m in min_samples_leaf:
				accuracy1 = 0
				for i in range(k):
					dt = DecisionTreeClassifier(max_depth=p , min_samples_split=c , min_samples_leaf=m)
					dt.fit(x_train,y_train)
					y_pred = dt.predict(x_test)
					accuracy1 = np.mean(y_test == y_pred)
					# print str(accuracy*100)+'%'
					parameters.append([p,c,m])
					accuracies.append(accuracy*100)
	print parameters
	print accuracies

	fig = plt.figure()
	plt.bar(np.arange(len(accuracies)),accuracies)
	plt.xticks(np.arange(len(accuracies)), ['param: '+str(x) for x in np.arange(len(accuracies))])
	# plt.show()

	fig.savefig(args.plots_save_dir.strip() + args.train_data.strip().split('/')[-1][:-3] +'_'+args.model_name +'.png')
	# define the grid here

	# do the grid search with k fold cross validation

	# model = DecisionTreeClassifier(  ...  )

	# save the best model and print the results
else:
	raise Exception("Invald Model name")