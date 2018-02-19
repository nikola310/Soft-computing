import pickle
import cv2
from sklearn.externals import joblib

PATH = 'mlp_model.pkl'

d = joblib.load(PATH)

img = cv2.imread('8.png', 0)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.resize(img, (8,8))

img = img.flatten()

print(d.predict(img.reshape(1, -1)))