import numpy as np
import math
import cv2
import vectors as v
import utils as u
import os
from number import Number
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from mnist import MNIST
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

def getFitSVM():
	putanja = os.getcwd()
	putanja  = putanja[:-5] + "\mnist-data\\"
	
	# print('---------------')
	# print(putanja)
	# print(putanja[:-5])
	# mndata = MNIST('../minst-data/')
	# print(putanja)
	mndata = MNIST(putanja)
	mndata.gz = True
	
	
	# digits = datasets.load_digits()
	
	params_svm = { "kernel": ['linear','rbf'] }
	clf = svm.SVC(degree=3, gamma='auto', probability=True)
	grid_svm = GridSearchCV(clf, params_svm)
	# grid_svm = RandomizedSearchCV(clf, params_svm, n_iter=2)
	
	# train = digits.data[:-359]
	# train_target = digits.target[:-359]
	train, train_target = mndata.load_training()
	train = train[:10]
	train_target = train_target[:10]
	train_target = np.fromiter(train_target, dtype=np.int)
	# print(train)
	print(type(train_target))
	# print('pre fita')
	# clf.fit(train, train_target)
	test, test_target = mndata.load_testing()
	test = test[:10]
	test_target = test_target[:10]
	test_target = np.fromiter(test_target, dtype=np.int)
	# print(len(train))
	# test = digits.data[-359:]
	# test_target = digits.target[-359:]
	# grid_svm.fit(digits.data, digits.target)
	# print(grid_svm.score(digits.data,digits.target))
	grid_svm.fit(train, train_target)
	print('Train set score: ' + str(grid_svm.score(train, train_target)))
	print('Test set score: ' + str(grid_svm.score(test, test_target)))
	# return clf
	return grid_svm

def main():
	clf = getFitSVM()
	
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
	# print(tgFiP)
	
	# Lista brojeva za sabrati
	to_add = dict()
	predict_num = 0
	
	while(cap.isOpened()):
		flag, frame = cap.read()
		if flag:
			img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
			
			img_t = img_bin
			_, contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			
			contours_of_interest = []
			for contour in contours:
				x,y,w,h = cv2.boundingRect(contour)
				if w >= 5 and w <= 25 and h >= 5 and h <= 25:
					if not(w == 5 and h == 5):
						contours_of_interest.append(contour)
			
			contours_of_interest = discardInnerRectangles(contours_of_interest)
			
			# Kreiranje brojeva i dodavanje u sadasnji recnik
			for contour in contours_of_interest:
				x,y,w,h = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
				
				center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
				cv2.putText(frame, ('Width: ' + str(w) + ' Height: ' + str(h)), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
			
			cv2.line(frame, (lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]), (0,255,0), 2)
			
			
			for contour in contours_of_interest:
				x,y,w,h = cv2.boundingRect(contour)
				
				# if h < 10:
					# continue
				# print('Width: ' + str(w) + ' Height: ' + str(h))
				center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2), 0)
				dist, pnt, = v.pnt2line(center, lineCoords[0], lineCoords[1])
				
				pnt = (int(pnt[0]), int(pnt[1]))
				center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
				
				if dist < 20 and dist > 15:
				# if dist < 18 and dist > 15:
					# cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
					
					k1, l1 = u.getLineEquation(center, O)
					
					tgFi = center[1]/(center[0]-O[0])
					# print(tgFi)
					'''
						ugao izmedju dve prave
						tgfi = (tgfi2 - tgfi1) / (1 + tgfi1*tgfi2)
					'''
					# tgFiAP = (tgFi - tgFiP)/(1 + tgFi*tgFiP)
					# cv2.putText(frame, str(tgFiAP), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
					# print(tgFiAP)
					# Broj se nalazi s desne strane (ispod linije)
					if tgFi < tgFiP:
						cropped = img_t[y:y+h, x:x+w]
						# small = cropped
						# small = cv2.bitwise_not(cropped)
						# small = cv2.resize(cropped, (8, 8))
						small = cv2.resize(cropped, (8, 8))
						small = cv2.bitwise_not(small)
						# print(tgFi)
						# print(dist)
						cv2.imshow('small', small)
						cv2.waitKey(0)
						
						small = small.flatten()
						small = np.transpose(small)
						# print('Prediction:', clf.predict(small.reshape(1, -1)))
						
						# del left_near_numbers[ret['id']]
						predict_num += 1
						# print(predict_num)
						
				
				cv2.line(frame, pnt, center, (255,255,0), 1)
				# cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
			
			
			cv2.imshow('frame', frame)
			# cv2.imshow('binary', img_bin)
			
		else:
			cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
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