#!/usr/bin/python
import numpy

def legendre(x, n):
	""" Returns the nth legendre polynomial of x """
	L = []
	for i in range(n+1):
		if i == 0: 
			L.append(1)
			continue
		if i == 1: 
			L.append(x)
			continue

		k = float(i)
		Li = (2*k-1)*x*L[i-1] / (k) - (k-1)*L[i-2] / (k)
		L.append(Li)
	
	
	return L[n]

			
def polynomial_transform(x, n):
	""" Returns the nth order legendre polynomial transform of x """
	x1 = x2 = 0
	z = []
	if len(x) == 3:
		x1 = x[1]
		x2 = x[2]
	if len(x) == 2:
		x1 = x[0]
		x2 = x[1]
	
	for i in range(n+1):
		for j in range(n+1):
			if (i+j) > n: continue
			z1 = legendre(x1, i)
			z2 = legendre(x2, j)
			z.append(z1*z2)
	
	return z

def regularized_linreg(Z, y, lamb):
	""" Returns the regularized weights estimated by linear regression """ 
	ZT = numpy.transpose(Z)
	ZTZ = numpy.dot(ZT, Z)
	L = lamb * numpy.identity(len(ZTZ))
	ZTZL = ZTZ + L
	inv = numpy.linalg.inv(ZTZL)
	invZT = numpy.dot(inv, ZT)
	wreg = numpy.dot(invZT, y)
	return wreg

def cross_val(Z, y, lamb):
	""" Returns the cross validation error for linear reg with weight decay """
	
	N = len(y)
	"""
	ZT = numpy.transpose(Z)
	ZTZ = numpy.dot(ZT, Z)
	L = lamb * numpy.identity(len(ZTZ))
	ZTZL = ZTZ + L
	inv = numpy.linalg.inv(ZTZL)
	Zinv = numpy.dot(Z, inv)
	H = numpy.dot(Zinv, ZT)
	y_hat = numpy.dot(H, y)
	"""
	wreg = regularized_linreg(Z, y, lamb)
	y_hat = numpy.sign(numpy.dot(Z, wreg))
	ZT = numpy.transpose(Z)
	ZTZ = numpy.dot(ZT, Z)
	L = lamb * numpy.identity(len(ZTZ))
	ZTZL = ZTZ + L
	inv = numpy.linalg.inv(ZTZL)
	pseudoinv = numpy.dot(inv, ZT)
	pseudoinvT = numpy.transpose(pseudoinv)

	H_ish = numpy.multiply(Z, pseudoinvT)
	H_diag = [sum(i) for i in H_ish]

	

	indiv_err = sum(((y_hat[i] - y[i])/(1-H_diag[i]))**2 for i in range(N))
	Ecv = indiv_err
	Ecv /= N

	return Ecv

def calc_err(Z, y, w):
	y_hat = numpy.sign(numpy.dot(Z, w))
	y_diff = [1 if i[0] != i[1] else 0 for i in zip(y, y_hat)]
	err = sum(y_diff)/float(len(y))
	return err








