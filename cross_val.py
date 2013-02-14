#!/usr/bin/python

class CV:
	def __init__(self, _model, _k_range, _data):
		self.model = _model
		self.k_range = _k_range
		self.data = _data
	
	def Ecv_min(self):
		Ecvs = [self.Ecv(k) for k in self.k_range]
		print Ecvs
		Ecv_min = min(Ecvs)
		best_k = self.k_range[Ecvs.index(Ecv_min)]

		return Ecv_min, best_k, Ecvs


	def Ecv(self, k):
		cv_model = self.model
		cv_model.set_k(k)
		print k
		err = 0
		for i in range(len(self.data)-1):
			loo_data = self.data[:]
			left_out = loo_data.pop(i)
			cv_model.populate(loo_data)
			
			cv_model.train()
			prediction = cv_model.classify(left_out[0])
			if prediction != left_out[1]:
				err += 1
			#print "CV", i, "/300, k=", k, cv_model.name()

		print "CV done for k at", k, "on model", cv_model.name()

		return err / float(len(self.data))
	
	def populate(self, _data):
		self.data = _data
	
	def set_model(self, _model):
		self.model = _model
	
	def set_krange(self, _k_range):
		self.k_range = _k_range
				


