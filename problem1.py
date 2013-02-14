#!/usr/bin/python
from ann import ANN
from math import tanh
import numpy
tanhprime = lambda x: (1 - x**2)
ident = lambda x: x
identprime = lambda x: 1 if isinstance(x, numpy.float64) else numpy.ones(len(x))

"""Problem 1, testing neural network"""
#The architecture of the network is defined entirely by the weights
#Each row represents the weights of the edges going from one node (in this case the 0th node of the 0th layer) to each node in the next layer (aside from the bias node)
weights = [
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
data = [([1,1],1)]

ann = ANN(tanh, tanhprime, weights)
ann.populate(data)
print "Numerical gradient", ann.num_grad()
print "Actual gradient", ann.calc_err_grad()[0]
ann.diagnostic()
ann.set_ident(True)
print "Numerical Gradient, ident", ann.num_grad()
print "Actual Gradient, ident", ann.calc_err_grad()[0]
ann.diagnostic()
ann.set_ident(False)
ann.set_lamb(0.01 / 300)
print "Numerical Gradient, lambda = 0.01/N", ann.num_grad()
print "Actual Gradient, lambda = 0.01/N", ann.calc_err_grad()[0]

