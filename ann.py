#!/usr/bin/python

import random
import math
import numpy.linalg as nl
import numpy
import datetime
import copy
import matplotlib.pyplot as plot

class ANN:
	def __init__(self, _transform_func, _transform_prime, _weights, _ident=False, _lamb=0):
		self.transform_func = _transform_func
		self.transform_prime = _transform_prime
		self.weights = _weights
		self.ident = _ident
		self.lamb = _lamb
		self.L = len(self.weights)
		self.outputs = [[] for i in range(self.L + 1)]
		self.inputs = [[] for i in range(self.L)]
		self.deltas = [[] for i in range(self.L)]

	"""
	How self.weights works:
	Each layer contains a matrix which represents edges from the ith node
	of the current layer to the jth node of the next layer
	"""
	
	def reset(self):
		self.outputs = [[] for i in range(self.L + 1)]
		self.inputs = [[] for i in range(self.L)]
		self.deltas = [[] for i in range(self.L)]
		return

	def populate(self, _data):
		self.data = _data

	def set_trans(self, _transform_func, _transform_prime):
		self.transform_func = _transform_func
		self.transform_prime = _transform_prime

	def set_ident(self, _ident):
		self.ident = _ident
	
	def set_lamb(self, _lamb):
		self.lamb = _lamb

	def set_weights(self, _weights):
		self.weights = _weights

	
	def initialize(self, value=None):
		for layer in self.weights:
			for node in layer:
				for weight in node:
					if value != None:
						weight = value
					else:
						weight = random.rand()

	def forward_prop(self, x):
		output = numpy.array(list(x))
		for l in range(self.L):
			bias_out = numpy.concatenate([[1], output])
			self.outputs[l] = bias_out
			layer = numpy.array(self.weights[l])
			wl_T = numpy.transpose(layer)
			s = numpy.dot(wl_T,bias_out)
			self.inputs[l] = s		
			theta = numpy.array(self.transform(s))
			if l == self.L-1 and self.ident:
				theta = numpy.array(s)
			output = theta
		self.outputs[-1] = output
		return output

	def back_prop(self):
		#Forward prop needs to be run first
		if not(self.ident):
			self.deltas[self.L-1] = self.transform_prime(self.outputs[-1])
		else:
			self.deltas[self.L-1] = [1]
		for l in reversed(range(self.L-1)):
			#Remove bias nodes
			unbias = self.outputs[l+1][1:]
			unbias_w = numpy.array(self.weights[l+1][1:])
			#Initialize size of each delta vector
			self.deltas[l] = [[] for i in range(len(unbias))]
			
			for i in range(len(self.deltas[l])):
				x = self.transform_prime(unbias[i])
				self.deltas[l][i] = x * numpy.dot(unbias_w[i],self.deltas[l+1])

		return

	def calc_err_grad(self):
		Ein = 0
		#G = numpy.zeros_like(self.weights)
		G = copy.deepcopy(self.weights)	
		for layer in range(len(G)):
			for i in range(len(G[layer])):
				for j in range(len(G[layer][i])):
					G[layer][i][j] = 0
		for i in self.data:
			x = i[0]
			self.reset()
			self.forward_prop(x)
			self.back_prop()
			Ein += (self.outputs[-1][0] - i[1])**2 / (4* len(self.data))
			if Ein > 10:
				print Ein
				print self.weights
				assert False
			for l in range(0,self.L):
				part1 = 2 * (self.outputs[-1][0] - i[1])

				unbias = numpy.transpose([self.outputs[l]])
				cur_delt = numpy.array([self.deltas[l]])
				part2 = (numpy.dot(unbias, cur_delt))
				Gn = part1 * part2
				G[l] += Gn / float(4 * (len(self.data)))
		if self.lamb != 0:
			for l in range(len(G)):
				for r in range(len(G[l])):
					for g in range(len(G[l][r])):
						G[l][r][g] += 2*self.lamb*self.weights[l][r][g]
			Ein += (self.lamb * sum([w**2 for l in self.weights for r in l for w in r]))
		return G, Ein

	def num_grad(self, epsilon=0.001):
		#Manually create a gradient of 0s in the same shape as the weights
		
		gradient = copy.deepcopy(self.weights)	
		for layer in range(len(gradient)):
			for i in range(len(gradient[layer])):
				for j in range(len(gradient[layer][i])):
					gradient[layer][i][j] = 0

		for layer in range(len(self.weights)):
			for i in range(len(self.weights[layer])):
				for j in range(len(self.weights[layer][i])):
					#Perturb weight
					self.weights[layer][i][j] += epsilon
					f1 = self.calc_err()
					self.weights[layer][i][j] -= 2*epsilon
					f2 = self.calc_err()
					#Return weight to original state
					self.weights[layer][i][j] += epsilon
					gradient[layer][i][j] = (f1 - f2) / (2 * epsilon) + float(2*self.lamb*self.weights[layer][i][j])
		return gradient

	def calc_err(self):
		err = 0
		for i in self.data:
			self.reset()
			self.forward_prop(i[0])
			result = self.outputs[-1][0]
			err += (result - i[1])**2
			if self.lamb != 0:
				err += (self.lamb * sum([w**2 for l in self.weights for r in l for w in r]))
		err /= float(4 * len(self.data)) 
		return err


	def transform(self, s):
		return [self.transform_func(node) for node in s]

	def diagnostic(self):
		#Prints inputs, outputs, sensitivities
		for i in range(len(self.weights)):
			print "x at layer:",i, "\t",self.outputs[i] 
			print "s at layer:",i+1, "\t",self.inputs[i] 
			print "delta at layer:",i, "\t",self.deltas[i] 

		print "final x:, x at layer: ", self.L, "\t", self.outputs[-1]

	def train(self, max_itr=100, eta=0.01, alpha=1.1, beta=0.8):
		#Train using variable learning rate gradient descent
		Ein = [self.calc_err_grad()[1]]
		weights_list = [self.weights]
		i = 1

		while i <= max_itr:
			err_grad = self.calc_err_grad()
			grad = -1 * (numpy.array(err_grad[0]))
			Ecur = err_grad[1]
			preserved = copy.deepcopy(self.weights)
			self.weights += eta*grad
			Epotential = self.calc_err_grad()[1]
			if Ecur > Epotential:
				#Accept
				weights_list.append(self.weights)
				eta = alpha*eta
				Ein.append(Epotential)
				"""
				if Ein[i-1] == Ein[i]:
					break
				"""
				i += 1
				continue
			else:
				#Reject
				#print eta,"rejected, using", beta*eta
				Ein.append(Ecur)
				i += 1
				self.weights = copy.deepcopy(preserved)
				eta = beta*eta
				continue
		return Ein, i
	
	def test_err(self, test_data):
		Etest = 0
		for i in test_data:
			y = i[1]
			x = i[0]
			h = numpy.sign(self.forward_prop(x))
			Etest += (1 / float((4* len(test_data)))) * ((h - y)**2)
		return Etest

	def classify(self, x):
		return numpy.sign(self.forward_prop(x))

	def decision(self, density=100, xlim=(-1.0,1.0), ylim=(-1.0,1.0), savename="ann_decisionbound.png", transfunc=None, overlay=None):
		""" Plots the decision boundary formed by the populated data """
		#Xlim and Ylim should be of the form (min, max)
		xlist = numpy.linspace(xlim[0], xlim[1], density)
		ylist = numpy.linspace(ylim[0], ylim[1], density)
		all_points = [(x,y) for x in xlist for y in ylist]
		
		pos = []
		neg = []
		for i in all_points:
			classification = 0
			if transfunc != None:
				z = transfunc(i)
				classification = self.classify(z)
			else:
				classification = self.classify(i)

			if classification == -1:
				neg.append(i)
			else:
				pos.append(i)

		if len(zip(*pos)) == 2:
			plot.scatter(zip(*pos)[0], zip(*pos)[1], marker='o', color='m')
		if len(zip(*neg)) == 2:
			plot.scatter(zip(*neg)[0], zip(*neg)[1], marker='o', color='g')
		

		"""Specific to problem 3.1, but an overlay of original data is necessary for transformed data"""

		pospoints = []
		negpoints = []
	
		for i in range(len(self.data)):
			x = ()
			classification = 0
			if overlay != None:
				x = overlay[i][0]
				classification = overlay[i][1]
			else:
				x = self.data[i][0]
				classification = self.data[i][1]

			if classification == -1:
				negpoints.append(x)
			else:
				pospoints.append(x)

		if len(zip(*pospoints)) == 2:
			plot.scatter(zip(*pospoints)[0], zip(*pospoints)[1], marker='o', color='b')
		if len(zip(*negpoints)) == 2:
			plot.scatter(zip(*negpoints)[0], zip(*negpoints)[1], marker='x', color='r')
		plot.savefig(savename)
		plot.clf()

		return
	
			
