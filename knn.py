#!/usr/bin/python

import numpy
import matplotlib.pyplot as plot
import sys

class KNN:
	
	"""Takes a data set, runs K-Nearest Neighbors, then plots the decision reason and saves it. This is currently for Binary Classification"""
	def __init__(self, k_=1, dist_type=0):
		self.k = k_
		self.dtype = dist_type
	
	def name(self):
		return "knn"
	
	def populate(self, data):
		self.data = data
	
	def classify(self, point):
		if isinstance(point[0], tuple):
			point = point[0]
		classification = 0
		temp = self.data[:]
		for i in range(len(temp)):
			if temp[i][0] == point:
				temp.pop(i)
				break
		def dist(point1):
			p1 = numpy.array(point)
			p2 = numpy.array(point1[0])
			diff = p1 - p2
			dist = numpy.linalg.norm(diff)
			return dist

		sorted_data = sorted(temp, key=dist)
		
		for i in range(self.k):
			classification += sorted_data[i][1]

		classification /= float(self.k)

		return numpy.sign(classification)

	def find_closest(self, exclude, point):
		closest_dist = sys.maxint
		best_index = 0
		for i in range(len(self.data)):
			if i in exclude: 
				continue
			dist = self.distance(point, self.data[i][0])
			if dist < closest_dist:
				closest_dist = dist
				best_index = i

		
		return best_index

	def distance(self, point, point1):
		""" Helping function to find distance between two points. """
		dist = 0
		p1 = numpy.array(point)
		p2 = numpy.array(point1)
		if self.dtype == 0:
			""" Euclidean Distance """
			diff = p1 - p2
			dist = numpy.linalg.norm(diff)

		if self.dtype == 1:
			""" Cos Similarity """
			denom = numpy.linalg.norm(p1) * numpy.linalg.norm(p2)
			dist = numpy.dot(p1, p2) / denom
			
		return dist

	def set_k(self, new_k):
		self.k = new_k

	def decision(self, density=100, xlim=(-1.0,1.0), ylim=(-1.0,1.0), savename="decisionbound.png", transfunc=None, overlay=None):
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
	
	def Etest(self, test_data):
		err = 0
		for i in test_data:
			predicted = self.classify(i[0])
			if predicted != i[1]:
				err += 1
		
		return err / float(len(test_data))



	def train(self):
		#Implemented because I hate OO stuff
		return






	
		

