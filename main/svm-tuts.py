import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn import datasets
from sklearn import svm
from PIL import Image

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001, C=100)

# print(len(digits.data))

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[-3].reshape(1, -1)))