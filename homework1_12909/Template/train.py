import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib
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

x = np.array(X)
y = np.array(Y)

y_temp = []
for i in y:
	for j in range(len(i)):
		if(i[j]==1):
			y_temp.append(j)
y_temp = np.array(y_temp)
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

# Train the models
if args.model_name == 'GaussianNB':
	nb = GaussianNB()
	nb.fit(x,y_temp)
	url = args.weights_path.strip() + args.train_data.strip().split('/')[-1][:-3]+'_'+args.model_name +'.pkl'
	joblib.dump(nb,url)

elif args.model_name == 'LogisticRegression':
	penalty = ['l1','l2']
	C = [0.5,1.0,1.5]
	max_iter = [100,150,200]
	parameters = []
	accuracies = []
	for p in penalty:
		for c in C:
			for m in max_iter:
				accuracy1 = 0
				for i in range(k):
					lr = LogisticRegression(penalty=p , C=c , max_iter=m)
					lr.fit(x_training[i],y_training[i])				
					y_pred = lr.predict(x_testing[i])
					accuracy1 += np.mean(y_testing[i] == y_pred)

				accuracy = float(accuracy1)/5
				parameters.append([p,c,m])
				accuracies.append(accuracy*100)
	# print parameters
	# print accuracies


	fig = plt.figure()
	plt.bar(np.arange(len(accuracies)),accuracies)

	fig.savefig(args.plots_save_dir.strip() + args.train_data.strip().split('/')[-1][:-3] +'_'+args.model_name +'.png')

	max_accuracy_index = np.argmax(np.array(accuracies))
	max_accuracy_param = parameters[max_accuracy_index]
	
	print max_accuracy_param
	
	lr = LogisticRegression(penalty = max_accuracy_param[0],C = max_accuracy_param[1],max_iter = max_accuracy_param[2],verbose = 0)
	lr.fit(x,y_temp)
	
	url = args.weights_path.strip() + args.train_data.strip().split('/')[-1][:-3]+'_'+args.model_name +'.pkl'
	joblib.dump(lr,url)

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
					dt.fit(x_training[i],y_training[i])
					y_pred = dt.predict(x_testing[i])
					accuracy1 += np.mean(y_testing[i] == y_pred)

				accuracy = float(accuracy1)/5
				parameters.append([p,c,m])
				accuracies.append(accuracy*100)

	fig = plt.figure()
	plt.bar(np.arange(len(accuracies)),accuracies)

	fig.savefig(args.plots_save_dir.strip() + args.train_data.strip().split('/')[-1][:-3] +'_'+args.model_name +'.png')


	max_accuracy_index = np.argmax(np.array(accuracies))
	max_accuracy_param = parameters[max_accuracy_index]
	
	print max_accuracy_param
	

	dt = DecisionTreeClassifier(max_depth = max_accuracy_param[0],min_samples_split = max_accuracy_param[1],min_samples_leaf = max_accuracy_param[2],max_features=None)
	dt.fit(x,y_temp)	
	url = args.weights_path.strip() + args.train_data.strip().split('/')[-1][:-3]+'_'+args.model_name +'.pkl'
	joblib.dump(dt,url)

else:
	raise Exception("Invald Model name")
