#!/usr/bin/python

import math
import numpy
import matplotlib.pyplot as plot
from kmc import KMC

class RBF:
	def __init__(self, _k=1):
		self.k = _k
		self.r = 2 / math.sqrt(self.k)
		self.centers = []
		self.weights = []
		self.update = True

	def name(self):
		return "rbf"

	def populate(self, _data):
		self.update = True
		self.data = _data

	def set_k(self, _k):
		self.k = _k
		self.r = 2 / math.sqrt(self.k)
		self.update = True
	
	def find_centers(self):
		kmc = KMC(self.k)
		kmc.populate(self.data)
		self.centers = kmc.lloyds()
		self.update = False
		return self.centers

	def get_centers(self):
		return self.centers
	
	def train(self):
		centers = []
		if self.update:
			centers = self.find_centers()
		else:
			centers = self.centers
		y = zip(*self.data)[1]
		theta = []
		for i in self.data:
			row = []
			for j in centers:
				thetaij = self.gaussian(self.scaled_dist(i,j))
				row.append(thetaij)
			theta.append(row)
		weights = numpy.dot(numpy.linalg.pinv(theta),y)
		self.weights = weights
		self.update = False

	def classify(self, x):
		weighted_sum = 0
		for i in range(self.k):
			mu = self.centers[i]
			w = self.weights[i]
			weighted_sum += w * self.gaussian(self.scaled_dist(x,mu))

		return numpy.sign(weighted_sum)

	def Etest(self, test_data):
		err = 0
		for i in test_data:
			predicted = self.classify(i[0])
			if predicted != i[1]:
				err += 1
		
		return err / float(len(test_data))

	def scaled_dist(self, x, mu):
		if isinstance(x[0], tuple):
			x = x[0]
		return self.dist(x,mu) / self.r

	def gaussian(self, x):
		return math.exp((x ** 2) / -2) / (2*math.pi)		
	
	def dist(self, x1, x2):
		x1 = numpy.array(x1)
		x2 = numpy.array(x2)
		diff = x1 - x2
		return numpy.linalg.norm(diff)
	
	def decision(self, density=100, xlim=(-1.0,1.0), ylim=(-1.0,1.0), savename="decisionbound_rbf.png", transfunc=None, overlay=None):
		""" Plots the decision boundary formed by the populated data """
		#Xlim and Ylim should be of the form (min, max)
		if self.update:
			print "WARNING: You should train me before plotting this boundary."
			self.train()
		
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


