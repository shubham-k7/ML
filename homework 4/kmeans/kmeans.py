import os
import os.path
import argparse
import csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics #adjusted_rand_score,adjusted_mutual_info_score,nomalized_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--k", type = int )
args = parser.parse_args()

def getData(filename):
	X = []
	Y = []
	with open(filename, 'rb') as fo:
		reader = csv.reader(fo)
		for row in reader:			
			X.append(map(float,row[:-1]))
			Y.append(float(row[len(row)-1]))

	random_indices=np.random.permutation(len(X))
	x = []
	y = []
	for i in range(len(random_indices)):
		x.append(X[random_indices[i]])
		y.append(Y[random_indices[i]])

	x = np.array(x)
	y = np.array(y)

	return x, y

def pltScatter(x,y,filename):
	fig = plt.figure()
	x_embedded = TSNE(n_components = 2,random_state = 1).fit_transform(x)
	plt.scatter(x_embedded[:,0],x_embedded[:,1],c=y,marker="o")
	fig.savefig('Plots/'+filename.strip())
	print "Saving Plot..."

def initMeans(x,k):
	maxIndex = x.shape[0]
	random_indices = np.random.choice(maxIndex,size = k, replace=False)
	means = []
	for i in range(k):
		means.append(x[random_indices[i]])
	
	return np.array(means)

def classify(means,item):

	index = -1
	min1 = sys.maxint
	for i in range(len(means)):
		dis = np.linalg.norm(means[i]-item)
		if(dis < min1):
			index = i
			min1 = dis
	return index

x,y = getData(args.data.strip())

k = args.k

ars = []
nmis = []
amis = []
for va in range(5):
	means = initMeans(x,k)

	belongsTo = [0 for i in range(len(x))]

	# print means

	maxIterations = 100
	dist = np.zeros((k,len(x)))
	# print dist
	noChange = True
	iterations = 0
	actviter = []
	while(noChange and iterations < maxIterations):
		iterations+=1
		meansCopy = means.copy()
		for j in range(len(meansCopy)):
			for i in range(len(x)):
				dist[j][i] = np.linalg.norm(x[i]-meansCopy[j])
			
		belongsTo = np.argmin(dist,axis = 0)

		means = []
		for i in range(len(meansCopy)):
			means.append(np.mean(x[belongsTo==i],axis = 0))
		means = np.array(means)

		np.sort(means,axis = 0)
		np.sort(meansCopy, axis = 0)

		val = []
		sum2 = 0
		for i in range(len(x)):
			sum2 += np.power(np.linalg.norm(x[i]-means[belongsTo[i]]),2)
		actviter.append(sum2)

		if(np.array_equal(means,meansCopy)):
			noChange = False
	ars.append(round(metrics.adjusted_rand_score(y,belongsTo),3))
	nmis.append(round(metrics.normalized_mutual_info_score(y,belongsTo),3))
	amis.append(round(metrics.adjusted_mutual_info_score(y,belongsTo),3))

print "Average ars ",np.mean(ars)
print "Average nmis",np.mean(nmis)
print "Average amis",np.mean(amis)
pltScatter(x,y,args.data.strip()+'_orig.png')
pltScatter(x,belongsTo,args.data.strip()+'_pred.png')
fig = plt.figure()
plt.plot(actviter)
print "Saving Plot..."
fig.savefig('Plots/' + args.data.strip()+'_ObjVsIter.png')