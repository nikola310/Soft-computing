import numpy as np
import cv2

def discardInnerRectangles(contours):
	to_delete = []
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
	
	return contours

def main():
	cap = cv2.VideoCapture('../data/video-1.avi')

	cv2.startWindowThread()

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
				if width > 4 and width < 30 and height > 3 and height < 30:
					contours_of_interest.append(contour)
			
			contours_of_interest = discardInnerRectangles(contours_of_interest)
			
			for contour in contours_of_interest:
				x,y,w,h = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
			
			cv2.imshow('frame', frame)
			cv2.imshow('binary', img_bin)
		else:
			# The next frame is not ready, so we try to read it again
			cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
			print("frame is not ready")
			# It is better to wait for a while for the next frame to be ready
			cv2.waitKey(1000)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		'''
			If the number of captured frames is equal
			to the total number of frames, we stop
		'''
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			break


	cap.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()