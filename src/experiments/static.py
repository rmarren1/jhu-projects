import util.clean_data as cd
import algos.similarity as sim
import algos.basic as basic
import pandas as pd
import cufflinks as cf
import numpy as np
import matplotlib.pyplot as plt
import algos.evaluate as evaluate
import seaborn as sns

def clust_corr(data, model):
	bl = block_rep(data, model)
	mean = np.mean(np.dstack(bl), axis=2)
	return mean

def block_rep(data, model):
	p = model['clust_params']
	p = [p] * len(data)
	corr = cd.run_function(model['correl'], data)
	clust = cd.run_function(model['clust'], zip(corr, p))
	bl = cd.run_function(sim.blocked, clust)
	return bl

def sep_test(evidence, labels, thresh):
	guesses = evidence
	guesses[guesses >= thresh] = 1
	guesses[guesses < thresh] = 2
	accuracy = evaluate.accuracy(guesses, labels)
	prec, rec = evaluate.prec_rec(guesses, labels)
	return accuracy, prec, rec

def thresh_simple(bl, labels, difference, thresh, plots = False):
	low, high = thresh
	diff = difference.copy()
	diff[np.abs(diff) < low] = 0
	diff[np.abs(diff) > high] = 0
	diff = diff - np.mean(diff)
	evidence = map(lambda x: np.sum((x * diff)), bl)
	evidence = np.array(evidence)
	if plots:
		ev_a = evidence[labels == 1]
		ev_c = evidence[labels == 2]
		band = 1
		sns.kdeplot(np.array(ev_a), bw = band, label='ASD')
		sns.kdeplot(np.array(ev_c), bw = band, label = 'Control')
		plt.title("Kernel Density Estimates of Classifier Scores.")
		plt.xlabel("Score Given by Classifier")
		plt.ylabel("Kernel Density Estimate")
		plt.show()
	out = sep_test(evidence, labels, 0)
	return out

def plot_evidence(ev, labels):
	ev_a = ev[labels == 1]
	ev_c = ev[labels == 2]
	plt.hist(ev_a, color='blue', alpha = .5, normed=1)
	plt.hist(ev_c, color='red', alpha = .5, normed=1)
	plt.show()
