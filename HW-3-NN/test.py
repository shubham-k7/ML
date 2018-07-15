#Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

scores = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(200):
  
  # evaluate class scores, [N x K]
  scores.append(np.dot(X, W[ni]) + b[ni]) 

print(len(scores))
  # compute the class probabilities
  # exp_scores = np.exp(scores)
  # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # # compute the loss: average cross-entropy loss and regularization
  # corect_logprobs = -np.log(probs[range(num_examples),y])
  # data_loss = np.sum(corect_logprobs)/num_examples
  # reg_loss = 0.5*reg*np.sum(W*W)
  # loss = data_loss + reg_loss
  # if i % 10 == 0:
  #   print "iteration %d: loss %f" % (i, loss)
  
  # # compute the gradient on scores
  # dscores = probs
  # dscores[range(num_examples),y] -= 1
  # dscores /= num_examples
  
  # # backpropate the gradient to the parameters (W,b)
  # dW = np.dot(X.T, dscores)
  # db = np.sum(dscores, axis=0, keepdims=True)
  
  # dW += reg*W # regularization gradient
  
  # # perform a parameter update
  # W += -step_size * dW
  # b += -step_size * db