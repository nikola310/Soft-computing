class Number:
	
	def __init__(self, contour, passed, small, tgFi=0, id=-1):
		self.contour = contour
		self.id = id
		self.passed = passed
		self.small = small
		self.tgFi = tgFi
	
	def __str__(self):
		return "Contour: {0}, Id: {1}, Passed: {2}".format(self.contour, self.id, self.passed)