import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )
parser.add_argument("--activate", type = str)
parser.add_argument("--model_dir", type = str)
args = parser.parse_args()

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

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
	for i in range(201):

		# forward pass 
		scores = [x_train]
		zs = []

		for li in range(n-1):
			
			num_examples = x_train.shape[0]
			
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


X,Y = load_h5py(args.data.strip())

x = np.array(X)
y = np.array(Y)

k = 2

x_temp = []

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

x = np.array(x1).astype(float)
y = np.array(y1)

x_sets = np.split(x[:-1],k)
y_sets = np.split(y[:-1],k)

x -= np.mean(x, axis = 0)

max1 = np.max(x) - np.min(x)
x = np.divide(x,max1)

best_accuracy = -1
n = 4
layer_sizes = [784,100,50,2]
mu = 0.95
reg = 0.001
step_size = 0.001
activations = args.activate.strip()

best_accuracy = -1

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
	x_test = np.array(x_test)
	y_test = np.array(y_test)

	print("For k = 1")
	model = NN(x_train,y_train,x_test,y_test,n,layer_sizes,activations,mu,reg,step_size)
	if(model["accuracy"] > best_accuracy):
		best_model = model
		best_accuracy = model["accuracy"]

print(best_model["accuracy"])
url = args.model_dir.strip() + args.data.strip()[-1][:-3]+'_best_model' +'.pkl'
joblib.dump(best_model,url)