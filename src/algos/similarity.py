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

def ap(x):
	x = -x
	af = AffinityPropagation(affinity='precomputed')
	af.fit(x)
	cluster_centers_indices = af.cluster_centers_indices_
	n_clusters_ = len(cluster_centers_indices)
	return n_clusters_
