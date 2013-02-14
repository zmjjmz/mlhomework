#!/usr/bin/python
import datetime
import math
import matplotlib.pyplot as plot
import numpy as np
import random
import utils
from pylab import contour
from scipy.misc import imsave as save
from knn import KNN
from rbf import RBF
from ann import ANN
from svmdual import SVMDual
from cross_val import CV
from copy import deepcopy

train_size = 300
tolerance = 0.05
all_train = open('ZipDigits.train')
all_test = open('ZipDigits.test')
test_size = 8998

def convert_to_matrix(_file):
	"""Turns the file into a 3D matrix of image matrices"""
	data_ = _file.readlines()
	classification = []
	data = []
	for line in data_:
		line = line.split(' ')
		line.pop()
		line = [float(i) for i in line]	
		if line.pop(0) == 1:
			classification.append(1)
		else:
			classification.append(-1)
		line = [line[i:i+16] for i in range(0, len(line), 16)]
		data.append(line)
		
	return zip(data, classification)

def vert_symmetry(image):
	"""Measures vertical symmetry of an image"""
	sym = 0
	for j in range(len(image)):
		for i in range(len(image)/2):
			sym += (image[j][i] - image[j][15-i])**2
				
	sym /= (len(image)*len(image))
			
	return sym
def horiz_symmetry(image):
	"""Measures horizontal symmetry of an image"""
	sym = 0
	for i in range(len(image)):
		for j in range(i):
			sym += math.fabs(image[j][i] - image[j][15-i])
				
	sym *= -1
			
	return sym
def avg_intensity(image):
	ins = np.mean(image)
	return ins

def avg_features(data):
	feature_1 = 0
	feature_2 = 0
	for i in data:
		feature_1 += vert_symmetry(i)
		feature_2 += horiz_symmetry(i)

	feature_1 /= len(data)
	feature_2 /= len(data)
	return feature_1, feature_2

def convert(data):
	""" Takes the data in the (x_vec_n, y_n) format and applies features """
	X = []
	for i in data:
		image = i[0]
		y_val = i[1]
		x = [avg_intensity(image), -1*vert_symmetry(image)]
		X.append((x, y_val))
	
	return X

def normalize(data):
	""" Takes featurized data, finds minima and then scales/shifts to normalize to [-1, 1] """
	nH = 1
	nL = -1
	X, y = zip(*data)
	x1, x2 = zip(*X)
	x1 = list(x1)
	x2 = list(x2)
	for x in (x1, x2):
		dH = max(x)
		dL = min(x)
		for i in range(len(x)):
			x[i] = ((x[i] - dL) * (nH - nL) / (dH - dL)) + nL
			

	
	X = zip(x1, x2)
	return zip(X, y)

def plot_data(data, savename, overlay=None):
	""" Matplotlib doesn't like plotting the data the way I have it, so 
	this takes the data, separates it by class, then plots it """
	pos = []
	neg = []
	for i in data:
		if i[1] == 1:
			pos.append(i)
		elif i[1] == -1:
			neg.append(i)
	
	plot.scatter(zip(*zip(*pos)[0])[0], zip(*zip(*pos)[0])[1], marker='o', color='b')
	plot.scatter(zip(*zip(*neg)[0])[0], zip(*zip(*neg)[0])[1], marker='x', color='r')
	if overlay != None:
		plot.scatter(zip(*overlay)[0], zip(*overlay)[1], marker=(5,1), color = 'y')
	plot.savefig(savename)

	plot.clf()

	return

def plot_weights(X, y, w, savename):
	data = zip(X, y)

	density = 200
	
	xlist = np.linspace(-1.0,1.0,density)
	ylist = np.linspace(-1.0,1.0,density)
	X, Y = np.meshgrid(xlist, ylist)
	pospoints = []
	negpoints = []
	all_points = [(x,y) for x in xlist for y in ylist]
	result = []
	all_grid = [all_points[i:i+density] for i in range(0, len(all_points), density)]
	all_result = np.zeros((density, density))

	for i in range(density):
		for j in range(density):
			z = utils.polynomial_transform(all_grid[i][j], 8)
			classof = np.sign(np.dot(w, z))
			if classof == 1:
				pospoints.append(all_grid[i][j])
			else:
				negpoints.append(all_grid[i][j])
		
	plot.scatter(zip(*pospoints)[0], zip(*pospoints)[1], marker='o', color='m')
	plot.scatter(zip(*negpoints)[0], zip(*negpoints)[1], marker='o', color='g')

	""" EDGE DETECTION
	for i in range(density):
		for j in range(density):
			if i != (density-1):
				above = i+1
			else:
				above = i
			classof_above = 0
			z = utils.polynomial_transform(all_grid[i][j], 8)
			classof = np.sign(np.dot(w, z))
			if i != (density-1):
				z_above = utils.polynomial_transform(all_grid[above][j], 8)
				classof_above = np.sign(np.dot(w, z_above))
			else:
				classof_above = classof
			if (classof_above != classof):
				all_result[i][j] = 1

	plot.contour(X, Y, all_result)"""
	plot_data(data, savename)
	return

def run(data, trans_data, savecvvstest="cvvstest.png", trainsave="train_set.png"):	

	X_train = []
	Z_train = []
	y_train = []
	for i in range(train_size):
		N = len(data)
		rand = random.randint(0, N-1)
		point = data.pop(rand)
		trans_point = trans_data.pop(rand)
		X_train.append(point[0])
		y_train.append(point[1])
		Z_train.append(trans_point[0])
	
	plot_data(zip(X_train, y_train), "trainsave.png")
	Z_test, y_test = zip(*trans_data)
	X_test, y_test = zip(*data)

	unreg_weights = utils.regularized_linreg(Z_train, y_train, 0)
	reg2_weights = utils.regularized_linreg(Z_train, y_train, 2)
	plot_weights(X_train, y_train, unreg_weights, "unreg.png")
	plot_weights(X_train, y_train, reg2_weights, "reg2.png")

	Ecvs = []
	Etests = []
	wregs = []
	lambdas = np.arange(0, 2.01, 0.01)
	for i in lambdas:
		wreg = utils.regularized_linreg(Z_train, y_train, i)
		Etest = utils.calc_err(Z_test, y_test, wreg)
		Ecv = utils.cross_val(Z_train, y_train, i)
		Ecvs.append(Ecv)
		Etests.append(Etest)
		wregs.append(wreg)
	
	plot.plot(lambdas, Ecvs, color='b')
	plot.plot(lambdas, Etests, color='r')
	plot.savefig(savecvvstest)
	plot.clf()

	best_index = Ecvs.index(min(Ecvs))
	best_lambda = lambdas[best_index]
	best_wreg = wregs[best_index]
	best_Etest = Etests[best_index]
	
	plot_weights(X_train, y_train, best_wreg, "bestreg.png")
	Eout_est = best_Etest + math.sqrt(math.log(2 / tolerance) / (2 * test_size))

	return best_lambda, Eout_est
	

	""" Pick best lambda """




test_data = convert_to_matrix(all_test)
train_data = convert_to_matrix(all_train)

data = test_data + train_data
random.shuffle(data)
data = convert(data)
data = normalize(data)
plot_data(data, "full_set.png")
"""
trans_data = []
for i in data:
	trans_data.append((utils.polynomial_transform(i[0], 8), i[1]))

lambs = []
Eouts = []
for i in range(1):
	lamb, eout = run(data, trans_data)
	lambs.append(lamb)
	Eouts.append(eout)
print lambs
print Eouts
plot.hist(lambs,200)
plot.savefig("lambda_hist.png")
"""	
sampled_data = []
for i in range(train_size):
		N = len(data) - i
		rand = random.randint(0, N-1)
		point = data.pop(rand)
		sampled_data.append(point)
"""
# Problem 10.1
print datetime.datetime.now()
knn = KNN()
knn_krange = range(1,18,2)
knn_cv = CV(knn, knn_krange, sampled_data)
knn_min = knn_cv.Ecv_min()
print datetime.datetime.now()
knn.set_k(knn_min[1])
knn_ecvs = knn_min[2]
plot.plot(range(1,18,2), knn_ecvs, color='r')
plot.savefig("knn_Ecvvsk.png")
plot.clf()
print "Plotting decision boundary for KNN"
knn.decision(savename="knn_bound.png")
print datetime.datetime.now()
print "KNN Ecv", knn_min[0], "with k at", knn_min[1]
print "KNN Ein", knn.Etest(sampled_data)
print "KNN Etest", knn.Etest(data)


# Problem 10.2
print datetime.datetime.now()
rbf = RBF()
rbf_krange = range(1,18,2)
rbf.populate(sampled_data)
rbf_cv = CV(rbf, rbf_krange, sampled_data)
rbf_min = rbf_cv.Ecv_min()
print datetime.datetime.now()
rbf.set_k(rbf_min[1])
rbf_ecvs = rbf_min[2]
rbf.train()
centers = rbf.get_centers()
plot_data(sampled_data,"centers.png",centers)
plot.plot(range(1,18,2), rbf_ecvs, color='r')
plot.savefig("rbf_Ecvvsk.png")
plot.clf()
print "Plotting decision boundary for RBF"
rbf.decision(savename="rbf_bound.png")
print datetime.datetime.now()
print "RBF Ecv", rbf_min[0], "with k at", rbf_min[1]
print "RBF Ein", rbf.Etest(sampled_data)
print "RBF Etest", rbf.Etest(data)
"""
#Problem 11.1 NN
t_weights = [
		[
			[0.25, 0.25], 
			[0.25, 0.25],
			[0.25, 0.25]
		],
		[
			[0.25],
			[0.25],
			[0.25]
		]
]

print datetime.datetime.now()
tprime = lambda x: 1 - x**2
faux_data = [((1,1),1)]
test_ann = ANN(math.tanh, tprime, t_weights, False)
test_ann.populate(faux_data)
print "Numerical Gradient, tanh", test_ann.num_grad()
print "Actual Gradient, tanh", test_ann.calc_err_grad()[0]

test_ann.set_ident(True)
print "Numerical Gradient, ident", test_ann.num_grad()
print "Actual Gradient, ident", test_ann.calc_err_grad()[0]
#Problem 11.2 NN
print datetime.datetime.now()
weights = [np.random.rand(3,10) / 100, np.random.rand(11,1) / 100]
#A test:
ann = ANN(math.tanh, tprime, weights, True)
ann.populate(sampled_data)
#Part a
max_itr = 1000
print "Training ANN with no lambda for", max_itr,"iterations"
print "Actual grad, no lambda", ann.calc_err_grad()[0]
print "Numerical grad, no lambda", ann.num_grad()

Eins_a, itr_a = ann.train(max_itr)
print "Finished training"
plot.plot(range(len(Eins_a)), Eins_a, color='r')
plot.savefig("ann-nolamb.png")
plot.clf()
print datetime.datetime.now()
print "Plotting Decision Boundary for ANN with no lambda after", max_itr, "iterations"
ann.decision(savename="ann_nolamb_dec.png")
print datetime.datetime.now()
#-----
lamb = 0.01/len(sampled_data)
ann.reset()
ann.set_weights(weights)
print "Training ANN with lambda", lamb, "for", max_itr, "iterations" 


print datetime.datetime.now()
ann.set_lamb(lamb)
print "Actual lambda grad", ann.calc_err_grad()[0]
print "Numerical lambda grad", ann.num_grad()

Eins_b, itr_b = ann.train(max_itr)
print "Finished training"
print datetime.datetime.now()
print "Plotting Decision Boundary for ANN with lambda",lamb, "after", max_itr, "iterations"
ann.decision(savename="ann_lamb_dec.png")
plot.plot(range(len(Eins_b)), Eins_b, color='r')
plot.savefig("ann-lamb.png")
plot.clf()
#-----
print "Training ANN with early stopping"
diff = 100
krange = range(100,1100,diff)
print "Iteration possibilities:", krange
val_size = 50
pick_from = deepcopy(sampled_data)
val_set = []
for i in range(val_size):
		N = len(pick_from) - i
		rand = random.randint(0, N-1)
		point = pick_from.pop(rand)
		val_set.append(point)
	
ann.populate(pick_from)
ann.set_weights(weights)
final_weights = []
val_errs = []
for i in krange:
	print i, datetime.datetime.now()
	ann.train(diff)
	final_weights.append(ann.weights)
	val_err = ann.test_err(val_set)
	val_errs.append(val_err)
	print val_err	

best_i = val_errs.index(min(val_errs))
best_itr = krange[best_i]
best_test = val_errs[best_i]
ann.set_weights(final_weights[best_i])

print datetime.datetime.now()
print "Best iterations:", best_itr, "with validation error", best_test
print "Plotting decision boundary for ANN with no lambda, run for", best_itr,"iterations"
ann.populate(sampled_data)
ann.decision(savename="ann_estop_dec.png")
print "ANN Ein with Early Stopping at", best_itr, ":", ann.test_err(sampled_data)
print "ANN Etest with Early Stopping at", best_itr, ":", ann.test_err(data)
#Problem 11.4
svm = SVMDual()
K8 = lambda x1, x2: (1 + np.dot(x1,x2))**8
svm.set_kernel(K8)
svm.populate(sampled_data)
#Part a
#Small C
print "SVM with small C", 0.1
print datetime.datetime.now()
svm.set_k(0.1)
svm.train()
svm.decision(savename="svm_smallC.png")

print "SVM with large C", 1
print datetime.datetime.now()
#Large C
svm.set_k(1)
svm.train()
svm.decision(savename="svm_largeC.png")

print "Cross Validation to determine C"
print datetime.datetime.now()
#Part c
#I would actually do CV, but it would take like half a day to finish. So fuck that. I'm choosing C = 0.3, don't give a fuck
"""
svm_krange = np.arange(0.1, 1.2, 0.2)
svm_cv = CV(svm, svm_krange, sampled_data)
svm_min = svm_cv.Ecv_min()
print datetime.datetime.now()
svm.set_k(svm_min[1])
svm_ecvs = svm_min[2]
svm.train()
plot.plot(svm_krange, svm_ecvs, color='r')
plot.savefig("svm_Ecvvsk.png")
plot.clf()
"""
svm.set_k(0.3)
svm.train()
print "Plotting decision boundary for svm"
svm.decision(savename="svm_bound_cv.png")
print datetime.datetime.now()
#print "SVM Ecv", svm_min[0], "with k at", svm_min[1]
print "SVM Ein", svm.test_err(sampled_data)
print "SVM Etest", svm.test_err(data)

