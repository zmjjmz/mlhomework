#!/usr/bin/python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plot




class SVMDual:

  def __init__(self):
    self.data = []
    self._a = None
    self._weights = None
    self._b = None
    self._C = None
    self._K = lambda x1, x2: np.dot(x1, x2)
#}}}

  def set_k(self, C):
    self._C = C

  def set_kernel(self, K):
    self._K = K


#{{{ # Load class with points
  def populate(self, _points):
    self.data = _points
#}}}

  def train(self, eps=0.002):
    self.run(self._K, self._C, eps)
    self.weights()

#{{{ Train SVM on points
  def run(self, K, C, epsilon=0.0001):
    n = len(self.data)
    x = [x for x, _ in self.data]
    y = [y for _, y in self.data]
    self._K = K
    augment = lambda x: np.concatenate((x,[1]))

    kernel = np.zeros(shape=(n, n))
    for i in range(n):
      for j in range(n):
        kernel[i][j] = K(augment(x[i]), augment(x[j]))

    eta = [1.0/kernel[k][k] for k in range(n)]
    t = 0
    A = [np.zeros(n)]

    while True:
    
      a = A[t].copy()
      
      for k in range(n):
        total  = 0.0
        for i in range(n):
          total += a[i] * y[i] * kernel[i][k]

        a[k] += eta[k] * (1 - y[k] * total)

        if a[k] < 0:
          a[k] = 0
        if a[k]  > C:
          a[k] = C

      A.append(a)

      t += 1

      error = np.linalg.norm(A[t] - A[t-1])
      #print('%4d | %f' % (t, error))

      if error <= epsilon:
        break

    self._a = A[-1]
#}}}


#{{{ Generate b value
  def weights(self):
    x = [np.array(x) for x, _ in self.data]
    y = [y for _, y in self.data]
    n = len(x)
    a = self._a
    K = self._K
    augment = lambda x: np.concatenate((x,[1]))

    nsv = 0
    for v in a:
      if v > 0:
        nsv += 1
    #print('NSV')
    #print(nsv)

    total1 = 0
    for i in range(n):
      if a[i] <= 0:
        continue
      total1 += y[i]

    total2 = 0
    for i in range(n):
      if a[i] <= 0:
        continue
      for j in range(n):
        if a[j] <= 0:
          continue

        xi = augment(x[i])
        xj = augment(x[j])
        # total2 += a[i] * y[i] * K(augment(x[i]), augment(x[j]))
        # print(K)
        total2 += a[i] * y[i] * K(xi, xj)

    b = (1.0/float(nsv)) * (total1 - total2)
    #print('B VALUE')
    #print(b)
    self._b = b
#}}}


#{{{ Classify a point
  def classify(self, point):
    x = [np.array(x) for x, _ in self.data]
    y = [y for _, y in self.data]
    n = len(x)
    a = self._a
    K = self._K
    b = self._b

    sign = lambda v: -1 if v < 0 else 1
    augment = lambda x: np.concatenate((x,[1]))
    # if self._weights is None:
      # self.weights()
    # return sign(np.dot(self._weights, augment(point)))

    total = 0
    for i in range(n):
      if a[i] <= 0:
        continue
      total += a[i] * y[i] * K(augment(x[i]),augment(point))

    return sign(total)
#}}}
  def test_err(self, test_data):
    Etest = 0
    for i in test_data:
      x = i[0]
      y = i[1]
      h = self.classify(x)
      if h != y:
        Etest += 1
    Etest /= float(len(test_data))
    return Etest


#}}}

  def decision(self, density=100, xlim=(-1.0,1.0), ylim=(-1.0,1.0), savename="ann_decisionbound.png", transfunc=None, overlay=None):
    """ Plots the decision boundary formed by the populated data """
    #Xlim and Ylim should be of the form (min, max)
    xlist = np.linspace(xlim[0], xlim[1], density)
    ylist = np.linspace(ylim[0], ylim[1], density)
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
  
      


# vim:tw=72:et:sw=2:ts=2:fdm=marker
