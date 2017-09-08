#import our necessary packages

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

#load our dataset
digits = datasets.load_digits()



#define our classifier
clf = svm.SVC(gamma=0.001,C=100)

#prepare our data*************

X,Y = digits.data[:-1], digits.target[:-1]
#train our classifier
clf.fit(X,Y)
#Try to predict some data
print("Prediction :" + clf.predict(digits.data[-1]))

plt.show(digits.images[-1],cmap=plt.cm.gray_r,interpolation="nearest")

plt.show()
