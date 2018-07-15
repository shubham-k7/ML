import os
import os.path
import argparse
import h5py
import numpy as np
from mnist import MNIST
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--activate", type = str)
parser.add_argument("--model_dir", type = str)
args = parser.parse_args()


def activate(z,activation):
	if(activation=="sigmoid"):
		return 1/(np.exp(np.multiply(-1,z)) + 1)
	elif(activation=="relu"):
		return np.maximum(0,z)
	else:
		return -1

def NN(x_train,y_train,x_test,y_test,n,layer_sizes,activations,mu,reg,step_size):
	
	model = {}
	#Train a Linear Classifier

	W = []
	b = []

	num_examples = x_train.shape[0]
	best_ac = -1
	
	# initialize parameters randomly
	for ni in range(n-1):
		W.append(np.random.randn(layer_sizes[ni],layer_sizes[ni+1])/np.sqrt(layer_sizes[ni]).astype(float))
		b.append(np.zeros((1,layer_sizes[ni+1])).astype(float))

	# velocity

	vel_w = []
	vel_b = []
	for ni in range(n-1):
		vel_w.append(np.zeros(np.shape(W[ni])).astype(float))
		vel_b.append(np.zeros(np.shape(b[ni])).astype(float))

	# gradient descent loop
	for i in range(351):

		# forward pass 
		scores = [x_train]
		zs = []

		for li in range(n-1):			
			
			z = np.dot(scores[li], W[li]) + b[li]
			zs.append(z)

			if(li==n-2): 
				exp_scores = np.exp(z)
				probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
			else:
				score = activate(z,activations)
				scores.append(score)

		# compute the loss: average cross-entropy loss and regularization
		corect_logprobs = -np.log(probs[range(num_examples),y_train])
		data_loss = np.sum(corect_logprobs)/num_examples
		reg_loss = 0

		for wi in W:
			reg_loss += 0.5*reg*np.sum(wi*wi)
		
		loss = data_loss + reg_loss
		if(i%10 == 0):
			print("Iteration %d: Loss %f" % (i, loss))

		dscores = probs
		dscores[range(num_examples),y_train] -= 1
		dscores /= num_examples

		dw = []
		db = []
		# Back
		# print(len(scores))
		for j in range(n-2,-1,-1):
			# compute the gradient on scores

			dw.insert(0,np.dot(scores[j].T, dscores)) #= np.dot(X.T, dscores)
	  		db.insert(0,np.sum(dscores, axis=0, keepdims=True))
			
	  		df = np.dot(dscores,W[j].T)

	  		if(activations == "relu"):
	  			df[scores[j] <= 0 ] = 0
	  		elif(activations == "sigmoid"):
	  			df = np.multiply(df,np.multiply(scores[j],(1-scores[j])))

	  		dscores = df

	  	for j in range(n-1):
	  		dw[j] += reg * W[j]

	 	# update
		for j in range(n-1):
			vel_w[j] = mu * vel_w[j] - step_size*dw[j]
			vel_b[j] = mu * vel_b[j] - step_size*db[j]
			W[j] = W[j] + vel_w[j]
			b[j] = b[j] + vel_b[j]

		if(i%10 == 0):
			pred = x_test
			# print("Length")
			# print(len(b))
			for yo in range(n-2):
				temp1 = pred
				pred = np.dot(temp1,W[yo]) + b[yo]
				if(activations == "sigmoid"):
					pred = 1/(np.exp(np.multiply(-1,pred))+1)
				elif(activations == "relu"):
					pred = np.maximum(0,pred)

			pred = np.dot(pred,W[len(W)-1]) + b[len(b)-1]
			ans = np.argmax(pred,axis = 1)

			ac = np.mean(ans == y_test)
			print("Accuracy: %f" % ac)
		if(ac > best_ac):
			best_w = W
			best_ac = ac
			best_b = b
	model["accuracy"] = best_ac
	model["W"] = best_w
	model["B"] = best_b
	return model

mndata = MNIST('samples')

x_train,y_train = mndata.load_training()
x_test,y_test = mndata.load_testing()

x_train = np.array(x_train).astype(float)
y_train = np.array(y_train)

x_test = np.array(x_test).astype(float)
y_test = np.array(y_test)

x_train -= np.mean(x_train, axis = 0)
max1 = np.max(x_train) - np.min(x_train)
x_train = np.divide(x_train,max1)

x_test -= np.mean(x_test, axis = 0)
max1 = np.max(x_test) - np.min(x_test)
x_test = np.divide(x_test,max1)

best_accuracy = -1
n = 4
layer_sizes = [784,100,50,10]
mu = 0.99
reg = 0.01
step_size = 1
activations = args.activate.strip()
best_accuracy = -1
	
model = NN(x_train,y_train,x_test,y_test,n,layer_sizes,activations,mu,reg,step_size)
print(model["accuracy"])
url = args.model_dir.strip() + args.activate.strip() +'_best_model' +'.pkl'
joblib.dump(model,url)