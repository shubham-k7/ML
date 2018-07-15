import os
import os.path
import argparse
import json
import numpy as np
import csv
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type = str )
parser.add_argument("--test_data", type = str )

args = parser.parse_args()

def get_xy_test(data):
	x_array = []
	for dic in data:
		x = ""
		for xi in dic["X"]:
			x += (str(xi)+' ')
		x_array.append(x.strip())
	return np.array(x_array)

def get_xy_train(data):
	x_array = []
	y_array = []
	for dic in data:
		x = ""
		for xi in dic["X"]:
			x += (str(xi)+' ')
		x_array.append(x.strip())
		y_array.append(dic["Y"])
	return np.array(x_array),np.array(y_array)

def load_json(filename):
	json_data = open(filename).read()
	return json.loads(json_data)

train_data = load_json(args.train_data.strip())
test_data = load_json(args.test_data.strip())
	
x_train,y_train = get_xy_train(train_data)
x_test = get_xy_test(test_data)

vectorizer = TfidfVectorizer(sublinear_tf=True,ngram_range=(0,3),token_pattern=r"\b\w+\b",binary=True)

x_train_vector = vectorizer.fit_transform(x_train)

clf = svm.LinearSVC(C=0.543)
clf.fit(x_train_vector,y_train)

# To improve accuracy, remove outliers
y_test = clf.predict(x_train_vector)
x_temp = []
y_temp = []
for i in range(len(y_test)):
	if(y_test[i]==y_train[i]):
		x_temp.append(x_train[i])
		y_temp.append(y_train[i])
# re train model using new set of training data

x_train_vector = vectorizer.fit_transform(np.array(x_temp))
clf.fit(x_train_vector,y_temp)

x_test_vector = vectorizer.transform(x_test)

predictions = clf.predict(x_test_vector)

with open("predict.csv", "wb") as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(("Id","Expected"))
	for i in range(len(predictions)):
		writer.writerow((i+1,predictions[i]))