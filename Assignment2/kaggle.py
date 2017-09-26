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

vectorizer = TfidfVectorizer(token_pattern=r"\b\w+\b",ngram_range=(0,3))

x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

clf = svm.LinearSVC(random_state=0,C=0.352)
clf.fit(x_train_vector,y_train)

predictions = clf.predict(x_test_vector)

with open("predict.csv", "wb") as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(("Id","Expected"))
	for i in range(len(predictions)):
		writer.writerow((i+1,predictions[i]))