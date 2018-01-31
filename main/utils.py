import cv2
import numpy as np

# Definisanje potrebnih konstanti i globalnih varijabli
minLineLength = 100
maxLineGap = 10

def getLineEquation(p1, p2):
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]
	k = (y2 - y1) / (x2 - x1)
	l = y1 - k*x1
	return (k,l)

# Prepoznavanje linije upotrebom Houghove transformacije
def detectLine(img_bin, frame):
	lineCoords = []
	lines = cv2.HoughLinesP(img_bin, 1, np.pi/180, 100, minLineLength, maxLineGap)
	if lines is not None:
		# print(lines)
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			lineCoords.append((x1,y1,0))
			lineCoords.append((x2,y2,0))
			
	# Koordinate pocetne i krajnje linije tacke
	return lineCoords