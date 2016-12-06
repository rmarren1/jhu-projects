# similarity.py
# Author: Ryan Marren
# Date: November 2016
import numpy as numpy
from cdtw import pydtw
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

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
		d = pydtw.dtw(x, y, pydtw.Settings(step='p0sym',
										window = 'palival',
										param=2.0,
										norm=False,
										compute_path=False
			))
		return d.get_dist()
	return distance_matrix(x, metric)

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

def n_clusts(x):
	return np.array(map(lambda y: np.unique(y).shape[0], x.T))

def avg_clust_size(x):
	return np.array(map(lambda y: np.mean(np.bincount(y)),x.T))

def max_clust_size(x):
	return np.array(map(lambda y: np.max(np.bincount(y)),x.T))

def min_clust_size(x):
	return np.array(map(lambda y: np.min(np.bincount(y)),x.T))

def ap(x):
	af = AffinityPropagation(affinity='precomputed')
	af.fit(x)
	labs = af.labels_
	#k = len(labs)
	#clusters = [[] for _ in range(np.max(labs) + 1)]
	#for i in range(k):
	#	clusters[labs[i]].append(i)
	return labs
