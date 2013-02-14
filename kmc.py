#!/usr/bin/python

import math
import numpy
import random

#k-means clustering

class KMC:
	def __init__(self, _k=1, _tol=1):
		self.k = _k
		self.tolerance = _tol

	def populate(self, _data):
		#strip classifications for convenience
		self.data = zip(*_data)[0]
	
	def set_k(self, _k):
		self.k = _k

	def lloyds(self):
		
		centers = []
		Ekm = []
		for i in range(self.k):
			#Initialize the centers by picking essentially random ones -- assuming that the data is randomly sampled
			centers.append(self.data[i])

		def dist(p1, p2):
			p1 = numpy.array(p1)
			p2 = numpy.array(p2)
			diff = p1 - p2
			return numpy.linalg.norm(diff)

		def calc_ekm(S):
			err = 0
			for mu in range(len(centers)):
				for x in S[mu]:
					err += (dist(x,centers[mu]) ** 2)

			return err

		#Ekm.append(calc_ekm(centers)) #initial Ekm

		#the 0th position in each partition is the center
		t = 0
		while True:
			S = []
			for s in range(self.k):
				S.append([])
			temp_data = list(self.data[:])
			"""
			for mui in range(self.k):
				for k in range(len(temp_data)):
					addto = True
					#Go through the data
					for muj in range(self.k):
						#Go through every other center
						if muj == mui: continue
						if dist(temp_data[k],centers[muj]) < dist(temp_data[k],centers[mui]):
							#If this is true, this point belongs to some other partition -- abort
							addto = False

							break
					#If the addto bit survived all that, then this point belongs to this center, so add it to the partition
					if addto: S[mui].append(temp_data[k])
			"""
			#Trying something else...
			for point in range(len(temp_data)):
				distances = []
				#Construct distances to each center
				for i in range(self.k):
					distances.append(dist(temp_data[point],centers[i]))
				closest_center = distances.index(min(distances))
				S[closest_center].append(temp_data[point])
				
			for i in range(self.k):
				new_center = numpy.sum(S[i],axis=0) / float(len(S[i]))
				centers[i] = new_center

			Ekm.append(calc_ekm(S))
			if t > 0:
				if math.fabs(Ekm[t] - Ekm[t-1]) < self.tolerance:
					break
			t += 1
		#print "\tCenters computed for", self.k, "in", t, "iterations with errors:", Ekm
		self.centers = centers
		return self.centers	
		
