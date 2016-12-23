# similarity.py
# Author: Ryan Marren
# Date: November 2016
import numpy as numpy
import numpy as np
from sklearn.cluster import AffinityPropagation, SpectralClustering
from sklearn import metrics
import cdtw
from cdtw import cdtw_sakoe_chiba
from basic import mean_center

def distance_matrix(x, metric):
	n = x.shape[0]
	dist_arr = np.zeros([n, n])
	for i in range(x.shape[0]):
		for j in range(i+1, x.shape[0]):
			val = metric(x[i], x[j])
			dist_arr[i, j] = val
			dist_arr[j, i] = val
	return dist_arr

def dtw(x):
	def metric(x, y):
		return cdtw_sakoe_chiba(x, y, 2)
	X = distance_matrix(x, metric)
	X = np.sqrt(X)
	X = X / np.max(X)
	X = X * 2
	X = -X + 1
	#print np.sum(X < 0), np.sum(X > 0), np.sum(X == 0)
	return X

def tsap(x):
	d = x.shape[0]
	T = x.shape[1]
	out = []
	for t in range(3, T):
		s = np.empty([d, d])
		for i in range(d):
			for j in range(d):
				v = 0
				v += np.abs(1 + x[i, t - 3] - x[j, t - 3]) * (1.0/2**3)
				v += np.abs(1 + x[i, t - 2] - x[j, t - 2]) * (1.0/2**2)
				v += np.abs(1 + x[i, t - 1] - x[j, t - 1]) * (1.0/2)
				v += np.abs(1 + x[i, t] - x[j, t])
				s[i][j] = v
		s = -s
		out.append(np.array(ap(s)).T)
	return (np.column_stack(out))

def blocked(x):
	arrs = []
	for i in range(len(x)):
		arrs.append(np.array(x == x[i]))
	return np.vstack(arrs)

def n_clusts(x):
	return np.array(map(lambda y: np.unique(y).shape[0], x.T))

def avg_clust_size(x):
	return np.array(map(lambda y: np.mean(np.bincount(y)),x.T))

def max_clust_size(x):
	return np.array(map(lambda y: np.max(np.bincount(y)),x.T))

def min_clust_size(x):
	return np.array(map(lambda y: np.min(np.bincount(y)),x.T))

def ap(X):
	x, params = X
	x[np.isnan(x)] = 0.0
	x[np.isinf(x)] = 0.0
	x = x - 1
	af = AffinityPropagation(**params)
	af.fit(-x)
	labs = af.labels_
	return labs

def spec(X):
	x, params = X
	p = params.copy()
	x[np.isnan(x)] = 0.0
	x[np.isinf(x)] = 0.0
	X = np.exp(- (-x + 1) ** 2 / (2. * p['gamma'] ** 2))
	p.pop('gamma', None)
	sc = SpectralClustering(**params)
	sc.fit(X)
	labs = sc.labels_
	return labs

def rando(X):
	x, _ = X
	labs = []
	for i in range(x.shape[0]):
		labs.append(np.random.randint(0, 15))
	return np.array(labs)