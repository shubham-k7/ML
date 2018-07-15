import os
import os.path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str )
parser.add_argument("--activate", type = str)
args = parser.parse_args()

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X,Y = load_h5py(args.data.strip())

x = np.array(X)
y = np.array(Y)

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
print(x.shape)
y = np.array(y_temp)

n = 4
layer_sizes = [784,100,50,2]

#Train a Linear Classifier

# initialize parameters randomly
W = []
b = []
		# for i in range(len(W)):
		# 	print(W[i].shape)

step_size = 1e-0
reg = 1e-3 # regularization strength

for ni in range(n-1):
	W.append(0.01 * np.random.randn(layer_sizes[ni],layer_sizes[ni+1]))
	b.append(np.zeros((1,layer_sizes[ni+1])))

# for i in range(len(W)):
# 	print(W[i].shape)

print(x[0].shape)
# gradient descent loop
for i in range(100):
	
	scores = [x]
	zs = []
	for li in range(n-1):
		num_examples = x.shape[0]
		print(num_examples)
		# for j in range()
		# evaluate class scores, [N x K]
		print("yolo")
		print(W[li].shape)
		print(b[li].shape)

		z = np.dot(scores[li], W[li]) + b[li]
		zs.append(z)
		# score =
		if(li==n-2) 
			exp_scores = np.exp(z)
			probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
		else:
			score = activate(z,activations)
			scores.append(score)

	# compute the loss: average cross-entropy loss and regularization
	corect_logprobs = -np.log(probs[range(num_examples),y])
	data_loss = np.sum(corect_logprobs)/num_examples
	reg_loss = 0

	for wi in W:
		reg_loss += 0.5*reg*np.sum(wi*wi)
	
	loss = data_loss + reg_loss
	if i % 10 == 0:
		print "iteration %d: loss %f" % (i, loss)

	dscores = probs
	dscores[range(num_examples),y] -= 1
	dscores /= num_examples

	dw = []
	db = []
	# Back
	for j in range(n-2,-1,-1):
		# compute the gradient on scores

		dw.insert(0,np.dot(scores[j-1].T, dscores)) #= np.dot(X.T, dscores)
  		db.insert(0,np.sum(dscores, axis=0, keepdims=True))
		
  		df = np.dot(dscores,W[j].T)
  		if(activations == "relu"):
  			dtemp[scores[j-1] <= 0 ] = 0
  		elif(activations == "sigmoid")
  			dtemp = np.multiply(dtemp,np.multiply(scores[j],(1-scores[j])))

  		dscores = dtemp

  	dw.insert(0,np.dot(x.T,dscores))
  	db.insert(0,np.sum(dscores, axis=0, keepdims=True))
  	
	# perform a parameter update
	W += -step_size * dW
	b += -step_size * db

	if(i%10 == 0):

def activate(z,activation):

