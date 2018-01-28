class Number:
	
	def __init__(self, contour, id, passed):
		self.contour = contour
		self.id = id
		self.passed = passed
	
	def __str__(self):
		return "Contour: {0}, Id: {1}, Passed: {2}".format(self.contour, self.id, self.passed)