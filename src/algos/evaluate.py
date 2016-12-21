import numpy as np

def accuracy(guesses, labels):
	correct = np.sum(np.equal(guesses, labels))
	return float(correct)/len(labels)

def ROC(guesses, labels):
	import numpy as np # Why is this needed?
	guesses = np.array(guesses)
	labels = np.array(labels)
	tp = np.sum((guesses == 1) & (labels == 1))
	fp = np.sum((guesses == 1) & (labels == 2))
	np = np.sum(guesses == 1)
	tpr = float(tp) / np
	fpr = float(fp) / np
	return tpr, fpr

def prec_rec(guesses, labels):
	import numpy as np # Why is this needed?
	guesses = np.array(guesses)
	labels = np.array(labels)
	tp = np.sum((guesses == 1) & (labels == 1))
	fp = np.sum((guesses == 1) & (labels == 2))
	tn = np.sum((guesses == 2) & (labels == 2))
	fn = np.sum((guesses == 2) & (labels == 1))
	prec = float(tp) / (tp + fp)
	rec = float(tp) / (tp + fn)
	return prec, rec