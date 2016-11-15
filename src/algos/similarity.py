# similarity.py
# Author: Ryan Marren
# Date: November 2016
import numpy as numpy
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np

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
		return fastdtw(x, y, dist=euclidean)[0]
	return distance_matrix(x, metric)