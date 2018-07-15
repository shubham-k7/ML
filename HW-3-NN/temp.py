[7:56 PM, 10/23/2017] +91 98912 00414: Normal paper pe print karva liyo
[7:56 PM, 10/23/2017] +91 98912 00414: Itna farak nahi padta :p
[7:56 PM, 10/23/2017] +91 98912 00414: ID card leke Jaana important hai
[6:34 PM, 10/24/2017] Aman Agarwal IIITD: Yar ek mail draft kar sakta Hai ?
Mai mail kar dunga
[6:34 PM, 10/24/2017] Aman Agarwal IIITD: Basic likh de. Mai final kar dunga
[2:38 AM, 10/25/2017] Aman Agarwal IIITD: MLPClassifier(activation=activation, alpha=reg, hidden_layer_sizes=layers, learning_rate='constant',
	       learning_rate_init=lr, max_iter=j, momentum=mu, random_state=1,
	        solver='lbfgs', verbose=True)
[2:38 AM, 10/25/2017] Aman Agarwal IIITD: layers = (100,50)
activation = "logistic"
lr = 0.01
reg = 0.01
mu = 0.95
iterations = 101
[2:40 AM, 10/25/2017] +91 98912 00414: import os
import os.path
import argparse
import h5py
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
import operator
import matplotlib.pyplot as plt
import matplotlib

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

x = np.array(x_temp)
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

max1 = np.max(x)
x = np.divide(x,max1)

x_training = []
y_training = []

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

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	
	clf = MLPClassifier(activation="logistic", alpha=0.01, hidden_layer_sizes=(100,50), learning_rate='constant',
	       learning_rate_init=0.01, max_iter=100, momentum=0.95, random_state=1,
	        solver='lbfgs', verbose=True)

	clf.fit(x_train,y_train)
	y_pred = clf.predict(x_test)
	accuracy = np.mean(y_test == y_pred)
	print("Accuracy: ",accuracy)
	accuracies += accuracy
[9:26 AM, 10/25/2017] +91 98912 00414: import os
import os.path
import argparse
from mnist import MNIST
import numpy as np
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = int )

args = parser.parse_args()

mndata = MNIST('samples')

x_train,y_train = mndata.load_training()
x_test,y_test = mndata.load_testing()

x_train = np.array(x_train).astype(float)
y_train = np.array(y_train).astype(float)

x_test = np.array(x_test).astype(float)
y_test = np.array(y_test).astype(float)

x_train -= np.mean(x_train, axis = 0)
max1 = np.max(x_train) - np.min(x_train)
x_train = np.divide(x_train,max1)

x_test -= np.mean(x_test, axis = 0)
max1 = np.max(x_test) - np.min(x_test)
x_test = np.divide(x_test,max1)

accuracies = 0

iterations = 100
best = -1
x_train = np.array(x_train)
y_train = np.array(y_train)

layers = [(800),(100,50),(100,75),(200,50,100)]

clf = MLPClassifier(activation="relu", alpha=0.01, hidden_layer_sizes=layers[args.model], learning_rate='constant',
       learning_rate_init=0.01, max_iter=iterations, momentum=0.95, random_state=1,
        solver='lbfgs', verbose=True)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
accuracy = np.mean(y_test == y_pred)
if(accuracy > best):
	best = accuracy
	best_model = clf
print("Accuracy: ",accuracy)

print("Best accuracy: ",best)
if(best_model != None):
	url = args.model_dir.strip() + args.data.strip()[-1][:-3]+'_best_model' +'.pkl'
	joblib.dump(best_model,url)
[10:59 AM, 10/25/2017] Aman Agarwal IIITD: '''
	Reference : http://cs231n.github.io/neural-networks-case-study/
'''

import numpy as np
import math

'''
	layers = [5,10,15] each element is the no. of neurons in the layer
	activations = [sigmoid,relu,maxout] each element represents the activation function of the corresponing layer
	epochs = no. of iterations
	X = input matrix
	Y = corresponding labels
	lr = learning rate
	reg = regularization coefficient - ridge regularization
	mu = momentum
'''

def train(layers,activations,X,Y,x,y,lr,reg,mu,file,epochs=10000,loss_func="softmax"):
	
	f1 = open(file[0],'a')

	# This will be our final model
	model = {}
	model["activations"] = activations
	model["loss"] = loss_func
	model["valid"]=1
	num_examples = X.shape[0]

	if(layers.shape[0]<2): # Basic check
		model["valid"]=-1
		return model

	np.random.seed(0)
	W = []
	B = []
	W_dash = []
	B_dash = []

	# Initializing the weights and biases
	for i in range(layers.shape[0]-1):
		if(i<layers.shape[0]-2 and activations[i]=="relu"):
			W.append(np.random.randn(layers[i],layers[i+1])/np.sqrt(layers[i]/2))
		else:
			W.append(np.random.randn(layers[i],layers[i+1])/np.sqrt(layers[i]))

		B.append(np.zeros((1,layers[i+1])).astype(float))
		W_dash.append(np.random.randn(layers[i],layers[i+1])/np.sqrt(layers[i]))
		B_dash.append(np.zeros((1,layers[i+1])).astype(float))

	# initializing velocity
	vW = []
	vB = []
	vW_dash = []
	vB_dash = []
	for i in range(layers.shape[0]-1):
		vW.append(np.zeros(np.shape(W[i])).astype(float))
		vB.append(np.zeros(np.shape(B[i])).astype(float))
		vW_dash.append(np.zeros(np.shape(W_dash[i])).astype(float))
		vB_dash.append(np.zeros(np.shape(B_dash[i])).astype(float))

	best_accuracy = 0
	# Batch gradient descent
	for i in xrange(epochs):

		#Forward pass
		Z = []
		Z_dash=[]
		A = []

		Z.append(X.dot(W[0])+B[0])
		Z_dash.append(X.dot(W_dash[0])+B_dash[0])
		if(layers.shape[0]>2):
			if(activations[0]=="sigmoid"):
				A.append(1/(1+np.exp(np.multiply(-1,Z[0]))))
			elif(activations[0]=="relu"):
				A.append(np.maximum(0,Z[0]))
			elif(activations[0]=="maxout"):
				A.append(np.maximum(Z_dash[0],Z[0]))
			else:
				model["valid"]=-1
				return model

		for j in range(1,layers.shape[0]-1):
			Z.append(A[j-1].dot(W[j])+B[j])
			Z_dash.append(A[j-1].dot(W_dash[j])+B_dash[j])
			if(j<layers.shape[0]-2):
				if(activations[j]=="sigmoid"):
					A.append(1/(1+np.exp(np.multiply(-1,Z[j]))))
				elif(activations[j]=="relu"):
					A.append(np.maximum(0,Z[j]))
				elif(activations[j]=="maxout"):
					A.append(np.maximum(Z_dash[j],Z[j]))
				else:
					model["valid"]=-1
					return model

		# loss calculation
		if(loss_func=="softmax"):
			ex = np.exp(Z[layers.shape[0]-2])
			probs = (ex/np.sum(ex, axis=1,keepdims=True))
			log_correct = -np.log(probs[range(num_examples),Y])
			loss = np.sum(log_correct)/num_examples

			# including regularization
			for j in range(layers.shape[0]-1):
				loss += 0.5*reg*np.sum(W[j]*W[j])

			if(i%5==0):
				print "Epochs: %d Loss:%f" % (i+1,loss)
				f1.write("Epochs: %d Loss:%f\n" % (i+1,loss))
		else:
			model["valid"]=-1
			return model


		# backward pass

		# last gradients from softmax
		if(loss_func=="softmax"):
			last_delta = probs
			last_delta[range(num_examples),Y] -= 1
			last_delta /= num_examples
		else:
			model["valid"]=-1
			return model

		dW = []
		dB = []
		dW_dash = []
		dB_dash = []


		for j in range(layers.shape[0]-2,0,-1):

			dW.insert(0,np.dot(A[j-1].T,last_delta))
			dB.insert(0,np.sum(last_delta, axis=0, keepdims=True))
			dW_dash.insert(0,np.dot(A[j-1].T,last_delta))
			dB_dash.insert(0,np.sum(last_delta, axis=0, keepdims=True))
			
			dtemp = np.dot(last_delta,W[j].T)
			dtemp_dash = np.dot(last_delta,W_dash[j].T)
			if(activations[j-1]=="relu"):
				dtemp[A[j-1] <= 0] = 0
			elif(activations[j-1]=="sigmoid"):
				dtemp = np.multiply(dtemp, np.multiply(A[j-1],1-A[j-1]))
			elif(activations[j-1]=="maxout"):
				dtemp = np.maximum(dtemp,dtemp_dash)# may be wrong 
			else:
				model["valid"]=-1
				return model
			last_delta = dtemp


		dW.insert(0,np.dot(X.T,last_delta))
		dB.insert(0,np.sum(last_delta, axis=0, keepdims=True))
		dW_dash.insert(0,np.dot(X.T,last_delta))
		dB_dash.insert(0,np.sum(last_delta, axis=0, keepdims=True))

		# regularization gradient
		for j in range(layers.shape[0]-1):
			dW[j] += reg * W[j]
			dW_dash[j] += reg * W_dash[j]

		# update
		for j in range(layers.shape[0]-1):
			vW[j] = mu * vW[j] - lr * dW[j]
			W[j] += vW[j]
			vB[j] = mu * vB[j] - lr * dB[j]
			B[j] += vB[j]
			vW_dash[j] = mu * vW_dash[j] - lr * dW_dash[j]
			W_dash[j] += vW_dash[j]
			vB_dash[j] = mu * vB_dash[j] - lr * dB_dash[j]
			B_dash[j] += vB_dash[j]

		model["W"] = W
		model["B"] = B
		model["W_dash"] = W_dash
		model["B_dash"] = B_dash
		if(i%5 == 0):
			ac = accuracy(model,x,y)
			f1.write('accuracy: %.2f' % ac)
			f1.write("\n")
		if(ac>best_accuracy):
			best_accuracy = ac
			best_W = W
			best_B = B
	model["accuracy"] = best_accuracy
	model["W"] = best_W
	model["B"] = best_B
	f1.write("\n----\n")
	f1.close()
	
	return model


'''
	model = dictionary of weights
	X = inputs to be predicted
'''
def accuracy(model,X,Y):

	W = model["W"]
	B = model["B"] 
	W_dash = model["W_dash"]
	B_dash  = model["B_dash"]
	activations = model["activations"]
	loss = model["loss"]
	
	prediction = X

	# forward pass
	for i in range(len(W)-1):
		temp = prediction
		prediction = np.dot(temp,W[i]) + B[i]
		if(activations[i]=="maxout"):
			prediction_dash = np.dot(temp,W_dash[i]) + B_dash[i]
		if(activations[i]=="relu"):
			prediction = np.maximum(0,prediction)
		elif(activations[i]=="sigmoid"):
			prediction = (1/(1+np.exp(np.multiply(-1,prediction))))
		elif(activations[i]=="maxout"):
			prediction = np.maximum(prediction,prediction_dash)
		else:
			return -1

	prediction = np.dot(prediction,W[len(W)-1]) + B[len(B)-1]
	predicted_class = np.argmax(prediction, axis=1)

	acc = (np.mean(predicted_class == Y))
	print ('accuracy: %.2f' % acc)

	return acc