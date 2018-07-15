import os
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