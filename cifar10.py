import cPickle
import numpy as np
import os

def load_CIFAR_file(filename):
	with open(filename, 'rb') as f:
		datadict = cPickle.load(f)
		X = np.array(datadict['data'])
		Y = np.array(datadict['labels'])
		return X, Y

def load_CIFAR10(ROOT):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
		X, Y = load_CIFAR_file(f)
		xs.append(X)
		ys.append(Y)    
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	Xte, Yte = load_CIFAR_file(os.path.join(ROOT, 'test_batch'))
	return Xtr, Ytr, Xte, Yte

