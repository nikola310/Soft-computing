import numpy as np
import math
import cv2
import vectors as v
import utils as u
from number import Number

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
			if ret < 0.5:
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
	
	numbers_past = dict()
	numbers_present = dict()
	contour_past = []
	numbers_to_add = dict()
	
	left_near_numbers = dict()
	right_near_numbers = dict()
	
	while(cap.isOpened()):
		flag, frame = cap.read()
		if flag:
			img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
			
			img, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			
			contours_of_interest = []
			for contour in contours:
				center, size, angle = cv2.minAreaRect(contour)
				width, height = size
				if width > 3 and width < 28 and height > 3 and height < 28:
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
			
			# for key, value in numbers_present.items():
				# x,y,w,h = cv2.boundingRect(value.contour)
				# center, _, _= cv2.minAreaRect(contour)
				# cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(frame, str(num.id), (int(center[0]), int(center[1])), font, 1, (200, 255, 255), 1, cv2.LINE_AA)
			
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
					
					tgFi = center[1]/(center[0]-O[0])
					# print('************************************')
					# print(tgFi)
					# print(tgFiP)
					# print('************************************')
					if tgFi > tgFiP:
						# print('smt')
						num = Number(contour, False)
						ret = checkIfNumberExisted(num, left_near_numbers)
						# print(flag)
						if not ret['flag']:
							num.id = getUniqueId()
							print('smt elif')
							left_near_numbers[num.id] = num
							print('Na frejmu broj ' + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ' lista ima ' + str(len(left_near_numbers)) + ' clanova')
							
					# if tgFi < tgFiP:
						
						# cv2.putText(frame, str(tgFi), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
					
				
				cv2.line(frame, pnt, center, (255,255,0), 1)
				# cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
			
			# print('======================================' + str(len(contour_past)))
			# if len(contour_past) > 0:
				# for i in range(len(contour_past)):
					# for j in range(len(contours_of_interest)):
						# if i != j:
						# center, _, _ = cv2.minAreaRect(contour_past[0])
						# center1, _, _ = cv2.minAreaRect(contours_of_interest[j])
						# print('Diff0: ' + str(center[0]-center1[0]))
						# print('Diff1: ' + str(center[1]-center1[1]))
						# ret = cv2.matchShapes(contour_past[0], contours_of_interest[j], 1, 0.0)
						# font = cv2.FONT_HERSHEY_SIMPLEX
						# cv2.putText(frame, str(ret), (int(center1[0]), int(center1[1])), font, 1, (200, 255, 255), 1, cv2.LINE_AA)
						# print(ret)
			# cv2.putText(frame, 'A', (50,10), font, 1, (200, 255, 255), 1, cv2.LINE_AA)
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


	cap.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()