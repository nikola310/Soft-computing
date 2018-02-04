import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from PIL import Image

digits = datasets.load_digits()

# clf = svm.SVC(gamma=0.0001, C=100)

print(len(digits.data))

# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.6, random_state=0)

x,y = digits.data[:-10], digits.target[:-10]
# clf.fit(X_train,y_train)

# print('Prediction:', clf.predict(X_test))
# print('Results: ', y_test)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, digits.data, digits.target, cv=5)
print(scores)
print(clf.get_params())