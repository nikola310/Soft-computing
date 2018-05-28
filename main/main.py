import numpy as np
import cv2
import vectors as v
import utils as u
import os
from keras import models
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from keras.datasets import mnist
#from mnist import MNIST
expand = 7

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

def expandImage(img):
    rows, cols = img.shape
    matrica = np.zeros((rows+2*expand, cols+2*expand))

    for i in range(expand,rows+expand):
        for j in range(expand,cols+expand):
            matrica[i,j] = img[i-expand,j-expand]
    
    return matrica
    

def predictNumber(model, cropped):
    rows, cols = cropped.shape
    #cropped_real = cropped
    cropped = expandImage(cropped)
    #cropped = cv2.GaussianBlur(cropped, (5, 5), 1)
    small = cv2.resize(cropped, (28, 28), interpolation = cv2.INTER_NEAREST)

    scaled = small / 255
    flattened = scaled.flatten()
    broj = np.reshape(flattened, (1, 784))
    cv2.imshow('scaled', scaled)
    cv2.waitKey(0)
    predicted_result = model.predict(broj)
    final_result = np.argmax(predicted_result)
    print('Prediction: ', final_result)
    return final_result
    
    
def getFitSVM():
    putanja = os.getcwd()
    putanja  = putanja[:-5] + "\mnist-data\\"
    
    # print('---------------')
    # print(putanja)
    # print(putanja[:-5])
    # mndata = MNIST('../minst-data/')
    # print(putanja)
    mndata = mnist.load_data() #MNIST(putanja)
    mndata.gz = True
    accuracies_rdf = []
    
    # digits = datasets.load_digits()
    
    params_svm = { "kernel": ['linear','rbf'] }
    clf = svm.SVC(degree=3, gamma='auto', probability=True)
    grid_svm = GridSearchCV(clf, params_svm, verbose=1, n_jobs=-1)
    # grid_svm = RandomizedSearchCV(clf, params_svm, n_iter=2, n_jobs=-1, verbose=1000)
    
    # train = digits.data[:-359]
    # train_target = digits.target[:-359]
    train, train_target = mndata.load_training()
    train = train[:2000]
    train_target = train_target[:2000]
    train_target = np.fromiter(train_target, dtype=np.int)
    # print(train)
    print(type(train_target))
    # print('pre fita')
    # clf.fit(train, train_target)
    test, test_target = mndata.load_testing()
    # test = test[]
    # test_target = test_target[:3000]
    test_target = np.fromiter(test_target, dtype=np.int)
    # print(len(train))
    # test = digits.data[-359:]
    # test_target = digits.target[-359:]
    # grid_svm.fit(digits.data, digits.target)
    # print(grid_svm.score(digits.data,digits.target))
    print('Poceo fit...')
    grid_svm.fit(train, train_target)
    acc_rdf = grid_svm.score(test,test_target)
    accuracies_rdf.append(acc_rdf*100)

    print("grid_rdf search accuracy: {:.2f}%".format(acc_rdf * 100))
    print("grid_rdf search best parameters: {}".format(grid_rdf.best_params_))
    print('Train set score: ' + str(grid_svm.score(train, train_target)))
    print('Test set score: ' + str(grid_svm.score(test, test_target)))
    # return clf
    return grid_svm

def main():
    if __name__ == '__main__':
        #clf = getFitSVM()
        model = models.load_model('testfile.h5')
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
        
        suma = 0
        
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
                    if w >= 9 and w <= 25 and h >= 9 and h <= 25: #w>=5
                        contours_of_interest.append(contour)
                
                contours_of_interest = discardInnerRectangles(contours_of_interest)
                
                for contour in contours_of_interest:
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),1)
                    
                    center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
                    cv2.putText(frame, ('Width: ' + str(w) + ' Height: ' + str(h)), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
                
                cv2.line(frame, (lineCoords[0][0], lineCoords[0][1]), (lineCoords[1][0], lineCoords[1][1]), (0,255,0), 2)
                
                
                for contour in contours_of_interest:
                    x,y,w,h = cv2.boundingRect(contour)
                    
                    #center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2), 0)
                    dot = (int(x), int(y), 0)
                    dist, pnt, = v.pnt2line(dot, lineCoords[0], lineCoords[1])
                    
                    pnt = (int(pnt[0]), int(pnt[1]))
                    center = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
                    
                    if dist < 20 and dist > 15:
                    # if dist < 18 and dist > 15:
                        # cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
                        
                        k1, l1 = u.getLineEquation(center, O)
                        
                        tgFi = center[1]/(center[0]-O[0])
                        '''
                            ugao izmedju dve prave
                            tgfi = (tgfi2 - tgfi1) / (1 + tgfi1*tgfi2)
                        '''
                        # tgFiAP = (tgFi - tgFiP)/(1 + tgFi*tgFiP)
                        # cv2.putText(frame, str(tgFiAP), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
                        # print(tgFiAP)
                        # Broj se nalazi s desne strane (ispod linije)
                        if tgFi < tgFiP:
                            cropped = img_t[y - expand:y + h + expand, x - expand:x + w + expand]
                            br = predictNumber(model, cropped)
                            print('Prediction: ', br)
                            suma += br
                    
                    cv2.line(frame, pnt, center, (255,255,0), 1)
                    # cv2.putText(frame, str(dist), center, font, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('frame', frame)
                # cv2.imshow('binary', img_bin)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Ako smo stigli do kraja videa, onda prekidamo petlju
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        
        
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()