from math import sqrt




#{{{ Cross-Validation Class
class CrossValidation:

	def __init__(self):
		self._points = None


#{{{ Load class with points
	def points(self, points):
		self._points = points
#}}}


#{{{ Run validation with a classifier callback
	def run(self, classifier):
		k = int(sqrt(len(self._points)))
		total = 0.0
		for i in range(k):
			subset = self._points[i*k:(i+1)*k]
			errors = 0
			for point in subset:
				if classifier(point[0]) != point[1]:
					errors += 1
					print('fail')
				else:
					print('pass')
			total += float(errors) / len(subset)
		return float(total) / k
#}}}

#}}}




# vim:tw=72:et:sw=2:ts=2:fdm=marker
