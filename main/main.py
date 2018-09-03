# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:30:04 2018

@author: Nikola
"""
import numpy as np
import cv2
import vectors as v
import utils as u
from keras import models
from number import Number
from scipy.spatial import distance
import math
#expand = 4 # -> cnn == 88.25
expand = 6 # -> mlp = 88.81

'''
    Odbacuje konture koje se nalaze unutar druge konture
'''
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

def expandImage(img):
    rows, cols = img.shape
    matrica = np.zeros((rows+2*expand, cols+2*expand))

    for i in range(expand,rows+expand):
        for j in range(expand,cols+expand):
            matrica[i,j] = img[i-expand,j-expand]
    
    return matrica
    

def predictNumber(model, cropped):
    rows, cols = cropped.shape
    cropped = expandImage(cropped)

    small = cv2.resize(cropped, (28, 28), interpolation = cv2.INTER_CUBIC) #INTER_NEAREST)
    scaled = small / 255

    # za mlp?
    flattened = scaled.flatten()
    broj = np.reshape(flattened, (1, 784))
    '''
    #za cnn
    broj = np.expand_dims(scaled, axis=2)
    broj = np.expand_dims(broj, axis=0)
    
    '''
    
    #cv2.imshow('scaled', scaled)
    #cv2.waitKey(0)

    predicted_result = model.predict(broj)
    final_result = np.argmax(predicted_result)
    print('Prediction: ', final_result)
    return final_result

'''
    Metoda koja pronalazi konture na slici, obrađuje ih i pronolazi brojeve.
'''
def getNumbers(img_bin):
    _, contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
    contours_of_interest = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w >= 5 and w <= 25 and h >= 9 and h <= 25: #w>=5
            contours_of_interest.append(contour)
                
    contours_of_interest = discardInnerRectangles(contours_of_interest)
                
    for contour in contours_of_interest:
        x,y,w,h = cv2.boundingRect(contour)
    brojevi = []
    
    for contour in contours_of_interest:
        x,y,w,h = cv2.boundingRect(contour)

        num = Number(x,y,w,h)
        brojevi.append(num)
            
    return brojevi

'''
    Pronalazi prethodni element iz niza koji najbolje odgovara
    prosledjenom broju.
'''
def findNumber(number, oldNumbers):
    cadidates = []

    for old in oldNumbers:
        dist = distance.euclidean((number.x + number.w, number.y + number.h), (old.x + old.w, old.y + old.h))

        if dist < 30:
            cadidates.append([dist, old])

    cadidates = sorted(cadidates, key=lambda x: x[0])   

    if len(cadidates) > 0:
        return cadidates[0][1]
    else:
        return None

'''
    Ažurira postojeće brojeve. Za svaki broj pokreće funkciju koja će
    pokušati naći najbliži broj iz prethodnog frejma. Ako nađe, onda se
    postojećem broju ažurira pozicija. Ako ne nađe, onda ga dodaje kao
    novi broj u listu.
'''
def updateNumbers(newNumbers, oldNumbers):
    for new in newNumbers:
        old = findNumber(new, oldNumbers)

        if old is None:
            oldNumbers.append(new)       
        else:
            old.updateCoords(new.x, new.y, new.w, new.h)

'''
    Metoda koja izbacuje brojeve koji više nisu od interesa (predaleko su).
'''
def removeFarAwayNumbers(numbers):
    retVal = []
    for num in numbers:
        if (num.x + num.w < 620) and (num.y + num.h < 470):
        #if not (num.get_bottom_right()[1] > 470 or num.get_bottom_right()[0] > 620):
            retVal.append(num)
    
    return retVal

'''
    Metoda koja iscrtava brojeve na frejm
'''
def iscrtajBrojeve(numbers, frame):
    for num in numbers:
        center = (int((num.x + (num.x + num.w)) / 2), int((num.y + (num.y + num.h)) / 2))
        cv2.putText(frame, ('Width: ' + str(num.w) + ' Height: ' + str(num.h)), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (num.x,num.y), (num.x+num.w,num.y+num.h),(0,255,255),1)


'''
    Proverava da li je određeni broje prošao ispod
    linije.
'''
def checkIfPassed(tgFiP, O, lineCoords, broj):
    dot = (int(broj.x), int(broj.y), 0)
    dist, pnt, = v.pnt2line(dot, lineCoords[0], lineCoords[1])
                    
    pnt = (int(pnt[0]), int(pnt[1]))
    center = (int((broj.x + (broj.x + broj.w)) / 2), int((broj.y + (broj.y + broj.h)) / 2))
    
    if dist < 20 and dist > 15:
        k1, l1 = u.getLineEquation(center, O)
        tgFi = center[1]/(center[0]-O[0])
        
        '''
            Ugao između dve prave
            tgfi = (tgfi2 - tgfi1) / (1 + tgfi1*tgfi2)
            
            Ako je tgFi < tgFiP, onda se broj nalazi ispod linije,
            što znači da je prošao ispod linije.
        '''
        if tgFi < tgFiP:
            return True
        else:
            return False
    else:
        return False

'''
    Metoda koja obrađuje određeni video i upisuje
    rezultat u out.txt
'''
def processVideo(video, file, model):
    cap = cv2.VideoCapture('../data/' + video)
    
    cv2.startWindowThread()
    
    flag, frame = cap.read()
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
    
    #img_bin
    lineCoords = u.detectLine(img_bin, frame)
    
    k,l = u.getLineEquation((lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]))
    
    '''
        'ishodište', tj. tačka u kojoj prava seče x osu
        u zavisnosti od nje će se računati tangens fi ugla 
        y = kx + l za y = 0
    '''
    O = (-l/k, 0)
    
    '''
    tgFi = y / x
    '''
    tgFiP = lineCoords[0][1] / (lineCoords[0][0] - O[0])
    
    suma = 0
    
    brojevi = []
    
    while(cap.isOpened()):
        flag, frame = cap.read()
        if flag:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
            
            
            cv2.line(frame, (lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]), (0,255,0), 2)
            
            numbers = getNumbers(img_bin)
            
            if len(numbers) > 0:
                updateNumbers(numbers, brojevi)
            
            '''
                Proverava da li je broj prošao ispod linije,
                ako jeste onda ga sabira
            '''
            for br in brojevi:
                if br.passed == False:
                    if checkIfPassed(tgFiP, O, lineCoords, br) == True:
                        br.setPassed(True)
                        cropped = img_bin[br.y - expand:br.y + br.h + expand, br.x - expand:br.x + br.w + expand]
                        rez = predictNumber(model, cropped)
                        suma += rez
                
            brojevi = removeFarAwayNumbers(brojevi)
            
            #iscrtajBrojeve(brojevi, frame)
            #cv2.putText(frame, ('x1: ' + str(lineCoords[0][0]) + ' y1: ' + str(lineCoords[0][1])), (lineCoords[0][0], lineCoords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, ('x2: ' + str(lineCoords[1][0]) + ' y2: ' + str(lineCoords[1][1])), (lineCoords[1][0], lineCoords[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
            #cv2.imshow(video, frame)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
        # Ako smo stigli do kraja videa, onda prekidamo petlju
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        
        
    cap.release()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print('suma -> ', suma)
    file.write('\n' + video + '\t' + str(suma))



def main():
    if __name__ == '__main__':
        
        model = models.load_model('model_mlp.h5') #model_cnn.h5
        
        #Otvaranje fajla za pisanje    
        out = open('out.txt', 'a')
        out.write('RA 13/2014 Nikola Stojanovic')
        out.write('\nfile\tsum')
        
        videos = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi', 'video-5.avi', 'video-6.avi', 'video-7.avi', 'video-8.avi', 'video-9.avi']
        print('Zapoceo obradu videa')
        for video in videos:
            print(video)
            processVideo(video, out, model)
        
        print('Kraj...')


main()