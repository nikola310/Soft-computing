import cv2
import numpy as np

# Definisanje potrebnih konstanti i globalnih varijabli
minLL = 80
maxLG = 100

def getLineEquation(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    k = np.float64((y2 - y1)) / (x2 - x1)
    l = y1 - k*x1
    return (k,l)

'''
    Prepoznavanje linije upotrebom Houghove transformacije
'''
def detectLine(img_bin, frame):
    lineCoords = []
    lines = cv2.HoughLinesP(img_bin, 1, np.pi/180, 180, 200, 150)
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            lineCoords.append((x1,y1,0))
            lineCoords.append((x2,y2,0))
    # Koordinate pocetne i krajnje tačke linije
    return lineCoords

'''
    Odredjuje pravu normalnu na prosledjenu pravu.
'''
def getPerpendicularLine(k1, l1, pnt):
    k2 = - 1 / k1
    l2 = pnt[1] - k2*pnt[0]
    return (k2, l2)

'''
    Vraca udaljenost izmedju dve tacke
'''
def getDistance(x1,y1,x2,y2):
    d = (x2 - x1)**2 + (y2 - y1)**2
    d = d**1/2
    return d