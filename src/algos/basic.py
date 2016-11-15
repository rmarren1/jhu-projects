# basic.py
# Author: Ryan Marren
# Date: November 2016
"""
Basic data analysis and preprocessing functions.

"""

import numpy as np

def mean_center(x):
	return x - np.mean(x, axis=1).reshape(-1, 1)

def noop(x):
	return x

def transpose(x):
	return x.T