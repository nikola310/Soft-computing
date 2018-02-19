import os
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.externals import joblib
from mnist import MNIST
from sklearn import datasets

'''
digits = datasets.load_digits()
clt = svm.SVC(gamma=0.01, C=100)
print(len(digits.data))
x,y = digits.data[:], digits.target[:]
clt.fit(x,y)
'''
PATH = 'mlp_model.pkl'

putanja = os.getcwd()
putanja  = putanja[:-5] + "\mnist-data\\"
# joblib.dump(clt, PATH)

# print('---------------')
# print(putanja)
# print(putanja[:-5])
# mndata = MNIST('../minst-data/')
# print(putanja)
mndata = MNIST(putanja)
mndata.gz = True
accuracies_rdf = []

# digits = datasets.load_digits()

params_svm = { "kernel": ['linear','rbf'] }
clf = svm.SVC(degree=3, gamma='auto', probability=True)
grid_svm = GridSearchCV(clf, params_svm, verbose=1)
# grid_svm = RandomizedSearchCV(clf, params_svm, n_iter=2, n_jobs=-1, verbose=1000)

# train = digits.data[:-359]
# train_target = digits.target[:-359]
train, train_target = mndata.load_training()
train = train[:200]
train_target = train_target[:200]
train_target = np.fromiter(train_target, dtype=np.int)


# clf.fit(train, train_target)
test, test_target = mndata.load_testing()
test = test[:300]
test_target = test_target[:300]
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
print("grid_rdf search best parameters: {}".format(grid_svm.best_params_))
print('Train set score: ' + str(grid_svm.score(train, train_target)))
print('Test set score: ' + str(grid_svm.score(test, test_target)))
joblib.dump(clf, PATH)
'''
# return clf
# return grid_svm