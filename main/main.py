import numpy as np
import cv2
from number import Number

# Definisanje potrebnih konstanti i globalnih varijabli
minLineLength = 100
maxLineGap = 10
id = -1 # identifikator svake konture

def discardInnerRectangles(contours):
	to_delete = []
	# print(len(contours))
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
	for k in range(len(to_delete)):
		del contours[k]
	# print(len(contours))
	return contours

# Prepoznavanje linije upotrebom Houghove transformacije
def detectLine(img_bin, frame):
	lineCoords = []
	lines = cv2.HoughLinesP(img_bin, 1, np.pi/180, 100, minLineLength, maxLineGap)
	if lines is not None:
		# print(lines)
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			lineCoords.append((x1,y1))
			lineCoords.append((x2,y2))
			
			# print(lineCoords)
	
	# Koordinate pocetne i krajnje linije tacke
	return lineCoords


def getUniqueId():
	global id 
	id += 1
	return id


def main():
	font = cv2.FONT_HERSHEY_SIMPLEX
	cap = cv2.VideoCapture('../data/video-0.avi')
	
	cv2.startWindowThread()
	
	flag, frame = cap.read()
	
	img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
	
	lineCoords = detectLine(img_bin, frame)
	
	
	numbers_past = []
	numbers_present = []
	
	while(cap.isOpened()):
		flag, frame = cap.read()
		if flag:
			img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			_, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
			
			# kernel = np.ones((3,3),np.uint8)
			# img_bin = cv2.erode(img_bin,kernel) #cv2.erode(img0,kernel)
			# img_bin = cv2.dilate(img_bin,kernel)
			# img_bin = cv2.dilate(img_bin,kernel)
			# cv2.imshow('dilate', img_bin)
			
			img, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			
			contours_of_interest = []
			for contour in contours:
				center, size, angle = cv2.minAreaRect(contour)
				
				width, height = size
				if width > 5 and width < 28 and height > 7 and height < 28:
					contours_of_interest.append(contour)
					
					# contours_present.append(contour)
					
					num = Number(contour, getUniqueId(), False)
					numbers_present.append(num)
			
			# print(len(contours_of_interest))
			contours_of_interest = discardInnerRectangles(contours_of_interest)
			# print(len(contours_of_interest))
			
			for contour in contours_of_interest:
				x,y,w,h = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
			
			
			# for i in range(len(contours_present)):
				# for j in range(len(contours_present)):
					# if i != j:
						# ret = cv2.matchShapes(contours_present[i], contours_present[j], 1, 0.0)
						# print(ret)
						# if ret < 0.0000001:
							# print('=======================')
							# print(i)
							# print('==')
							# print(j)
							# print('iste konture')
							# print('=======================')
			
			if len(numbers_past) > 0:
				# print(len(numbers_past))
				for numA in numbers_past:
					# print(numA)
					for numB in numbers_present:
						# print(numB)
						ret = cv2.matchShapes(numA.contour, numB.contour, 1, 0.0)
						print(ret)
						if ret < 0.001:
							print('=======================')
							print(numA.id)
							print('==')
							print(numB.id)
							print('iste konture')
							print('=======================')
							numB.id = numA.id
			
			cv2.line(frame, lineCoords[0], lineCoords[1], (0,255,0), 2)
			
			cv2.imshow('frame', frame)
			cv2.imshow('binary', img_bin)
			
			# for num in numbers_present:
				# print(num)
			
			del numbers_past[:]
			numbers_past = numbers_present
			del numbers_present[:]
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