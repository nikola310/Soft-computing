import numpy as np
import math
import cv2
import vectors as v
import utils as u
from number import Number
from sklearn import datasets
from sklearn import svm

id = -1 # identifikator svake konture

# Odbacuje konture koje se nalaze unutar druge konture
def discardInnerRectangles(contours):
	to_delete = []
	retVal = []
	for i in range(len(contours)):
		for j in range(len(contours)):
			if i != j:
				x1,y1,w1,h1 = cv2.boundingRect(contours[i])
				x2,y2,w2,h2 = cv2.boundingRect(contours[j])
				
				x11 = x1 + w1
				y11 = y1 + h1
				x22 = x2 + w2
				y22 = y2 + h2
				if ((x1 <= x22) and (x22 <= x11)):
					if ((y1 <= y22) and (y22 <= y11)):
						to_delete.append(j)
	
	for i in range(len(contours)):
		if not (i in to_delete):
			retVal.append(contours[i])
	return retVal

# Proverava da li je broj postojao u prethodnom frejmu
def checkIfNumberExisted(number, numbers_past):
	retFlag = False
	retNumId = -1
	if len(numbers_past) > 0:
		for key, value in numbers_past.items():
			ret = cv2.matchShapes(value.contour, number.contour, 1, 0.0)
			center, _, _ = cv2.minAreaRect(value.contour)
			centerN, _, _ = cv2.minAreaRect(number.contour)
			if ret < 0.1:
				retFlag = True
				retNumId = key
				break
	else:
		return {'flag' : retFlag, 'id' : retNumId}
	return {'flag' : retFlag, 'id' : retNumId}

def getUniqueId():
	global id 
	id += 1
	return id

def main():
	# Deo za masinsko ucenje
	digits = datasets.load_digits()
	clf = svm.SVC(gamma=0.001, C=100)
	x,y = digits.data[:], digits.target[:]
	clf.fit(x,y)
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cap = cv2.VideoCapture('../data/video-1.avi')
	
	cv2.startWindowThread()
	
	flag, frame = cap.read()
	
	img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
	
	lineCoords = u.detectLine(img_bin, frame)
	
	k,l = u.getLineEquation((lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]))
	
	'''
		'ishodiste', tj. tacka u kojoj prava sece x osu
		u zavisnosti od nje ce se racunati tangens fi ugla 
		y = kx + l za y = 0
	'''
	O = (-l/k, 0)
	
	'''
		tgFi = y / x
	'''
	tgFiP = lineCoords[0][1] / (lineCoords[0][0] - O[0])
	print(tgFiP)
	numbers_past = dict()
	numbers_present = dict()
	contour_past = []
	numbers_to_add = dict()
	
	left_near_numbers = dict()
	right_near_numbers = dict()
	
	# Lista brojeva za sabrati
	to_add = dict()
	predict_num = 0
	
	while(cap.isOpened()):
		flag, frame = cap.read()
		if flag:
			img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
			
			img_t = img_bin
			img, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			
			contours_of_interest = []
			for contour in contours:
				center, size, angle = cv2.minAreaRect(contour)
				width, height = size
				if width >= 3.5 and width < 28 and height >= 3.5 and height < 28:
					contours_of_interest.append(contour)
				
			
			contours_of_interest = discardInnerRectangles(contours_of_interest)
			
			# Kreiranje brojeva i dodavanje u sadasnji recnik
			for contour in contours_of_interest:
				num = Number(contour=contour, passed=False)
				x,y,w,h = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
				retDic = checkIfNumberExisted(num, numbers_past)
				if not retDic['flag']:
					# print('not retDic')
					num.id = getUniqueId()
				else:
					# print('else')
					num.id = retDic['id']
				numbers_present[num.id] = num
			
			cv2.line(frame, (lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]), (0,255,0), 2)
			
			numbers_past.clear()
			numbers_past = numbers_present
			numbers_present.clear()
			
			for contour in contours_of_interest:
				center, size, angle = cv2.minAreaRect(contour)
				center = (center[0], center[1], 0)
				x,y,w,h = cv2.boundingRect(contour)
				dist, pnt, = v.pnt2line(center, lineCoords[0], lineCoords[1])
				# dist, pnt, = v.pnt2line((x+w, y+h, 0), lineCoords[0], lineCoords[1])
				# dist, pnt, = v.pnt2line((x, y, 0), lineCoords[0], lineCoords[1])
				# print('Dist: ' + str(dist))
				pnt = (int(pnt[0]), int(pnt[1]))
				center = (int(center[0]), int(center[1])) #(x+w, y+h)
				
				if dist < 20:
					# cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
					
					k1, l1 = u.getLineEquation(center, O)
					
					cropped = img_t[y:y+h, x:x+w]
					# cv2.imshow('cropped', cropped)
					# cv2.waitKey(0)
					tgFi = center[1]/(center[0]-O[0])
					num = Number(contour, False, tgFi)
					# print(tgFi)
					'''
						ugao izmedju dve prave
						tgfi = (tgfi2 - tgfi1) / (1 + tgfi1*tgfi2)
					'''
					tgFiAP = (tgFi - tgFiP)/(1 + tgFi*tgFiP)
					# cv2.putText(frame, str(tgFiAP), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
					# print(tgFiAP)
					# Broj se nalazi s leve strane (iznad linije)
					if tgFi > tgFiP:
						
						
						ret = checkIfNumberExisted(num, left_near_numbers)
						if not ret['flag']:
							num.id = getUniqueId()
							# print('smt elif')
							left_near_numbers[num.id] = num
							# print('Na frejmu broj ' + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ' leva lista ima ' + str(len(left_near_numbers)) + ' clanova')
					
					# Broj se nalazi s desne strane (ispod linije)
					if tgFi < tgFiP:
						# ret = checkIfNumberExisted(num, right_near_numbers)
						ret = checkIfNumberExisted(num, left_near_numbers)
						
						if ret['flag']:
							# Znaci, ovaj broj je vec bio na levoj strani
							to_add[ret['id']] = left_near_numbers[ret['id']]
							
							cropped = img_t[y:y+h, x:x+w]
							small = cv2.resize(cropped, (64, 64))
							# print(small)
							small = cv2.bitwise_not(small)
							# print('-----------------------')
							small.flatten()
							# print(small)
							# cv2.imshow('cropped', cropped)
							# cv2.imshow('small', small)
							# cv2.imshow('contour', left_near_numbers[ret['id']].contour)
							# cv2.waitKey(0)
							print('Prediction:', clf.predict(small))
							
							del left_near_numbers[ret['id']]
							# print('to add=====================================')
							# print(len(to_add))
							predict_num += 1
							print(predict_num)
				
				cv2.line(frame, pnt, center, (255,255,0), 1)
				# cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
			
			
			cv2.imshow('frame', frame)
			# cv2.imshow('binary', img_bin)
			
			del contour_past[:]
			contour_past = contours_of_interest
		else:
			cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
			print("Frejm nije spreman")
			cv2.waitKey(1000)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		# Ako smo stigli do kraja videa, onda prekidamo petlju
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			break
	
	# if len(to_add)>0:
	# print(len(to_add))
		# for e in to_add:
			# print(e)
	
	cap.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()