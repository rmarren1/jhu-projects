# basic.py
# Author: Ryan Marren
# Date: November 2016
"""
Basic data analysis and preprocessing functions.

"""

import numpy as np
from sklearn.metrics import silhouette_score

def mean_center(x):
	return x - np.mean(x, axis=1).reshape(-1, 1)

def noop(x):
	return x

def transpose(x):
	return x.T

def _cluster_sort(x, c):
	ind = [item for sublist in c for item in sublist]
	x = x[ind]
	return x

def cluster_sort(x):
	return _cluster_sort(*x)

def correl(x):
	return np.corrcoef(x)

def _SSE(x, c):
	clust_sses = []
	for clust in c:
		data = x[clust]
		sse = np.sum(np.square(data))
		clust_sses.append(sse)
	return clust_sses

def SSE(x):
	return _SSE(*x)

def _silhouette(x, c):
	inds = [0] * x.shape[0]
	for i in xrange(len(c)):
		for j in xrange(len(c[i])):
			inds[c[i][j]] = i
	return silhouette_score(x, inds, metric='precomputed')

def silhouette(x):
	return _silhouette(*x)

def neg(x):
	return -x