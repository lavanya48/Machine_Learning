#loading datasets
from sklearn import datasets
iris= datasets.load_iris()
digits= datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images[1])

#learning and predicting fit(x,y) and predict(x,y)
from sklearn import svm
clf= svm.SVC(gamma=0.001, C=100)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))

#model persistence
from sklearn import datasets, svm
clf= svm.SVC(gamma='scale')
iris= datasets.load_iris()
x,y = iris.data, iris.target
clf.fit(x,y)
import pickle
s=pickle.dumps(clf)
clf2=pickle.loads(s)
clf2.predict(X[0:1])


#refitting and updating parameters
from sklearn import svm, datasets
import numpy as np
X, y= datasets.load_iris(return_X_y=True)
clf=svm.SVC()
clf.set_params(kernel='linear').fit(X,y)
print(clf.predict(X[:5]))
clf.set_params(kernel='rbf', gamma='scale').fit(X,y)
print(clf.predict(X[:5]))

#classifying irises knn algorithm
import numpy as np
from sklearn import datasets
iris= datasets.load_iris()
iris_X=iris.data
iris_y=iris.target

np.random.seed(0)
indices=np.random.permutation(len(iris_X))
iris_X_train= iris_X[indices[:-10]]
iris_y_train= iris_y[indices[:-10]]
iris_X_test= iris_X[indices[-10:]]
iris_y_test= iris_y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print(knn.predict(iris_X_test))
print(iris_y_test)






