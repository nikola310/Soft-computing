class Number:
    
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.passed = False
       
    def getPassed(self):
        return self.passed
    
    def setPassed(self, passed):
        self.passed = passed
        
    def __str__(self):
        return "X: {0}, Y: {1}, W: {2}, H: {3}".format(self.x, self.y, self.w, self.h)
    
    def updateCoords(self, x, y, w, h):
      self.x = x
      self.y = y
      self.w = w
      self.h = h
