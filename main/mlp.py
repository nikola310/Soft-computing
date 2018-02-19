import numpy as np
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
import os.path

'''
train, train_target = mndata.load_training()
 a = train[1]
 > pixels = np.array(a, dtype='uint8')
  pixels = pixels.reshape((28,28))
  plt.imshow(pixels, cmap='gray')
'''

PATH = 'mlp_model.pkl'

if __name__ == '__main__':
	print('Fetching and loading MNIST data')
	mnist = fetch_mldata('MNIST original')
	X, y = mnist.data, mnist.target
	X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

	print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
	print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

	clf = None
	if os.path.exists(PATH):
		print('Loading model from file.')
		clf = joblib.load(PATH).best_estimator_
	else:
		print('Training model.')
		params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
		mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
		clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
		clf.fit(X_train, y_train)
		print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
		print('Best params appeared to be', clf.best_params_)
		joblib.dump(clf, PATH)
		clf = clf.best_estimator_

		print('Test accuracy:', clf.score(X_test, y_test))
    
'''
Output:
Fetching and loading MNIST data
Got MNIST with 52500 training- and 17500 test samples
Digit distribution in whole dataset: [6903 7877 6990 7141 6824 6313 6876 7293 6825 6958]
Training model.
Finished with grid search with best mean cross-validated score: 0.97820952381
Best params appeared to be {'hidden_layer_sizes': (512,)}
Test accuracy: 0.982057142857
'''