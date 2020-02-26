#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:22:59 2019

@author: z003zhj
"""

import  matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics

digits= datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index,(image,label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index+1)
    plt.axis('off')
    plt.imshow(image, cmap= plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    
    
#to flatten the image, to turn the data in (samples, feature) matrix
n = len(digits.images)
data = digits.images.reshape((n, -1))
    
#fitting the classifier
clf= svm.SVC(gamma=0.001)
clf.fit(data[:n//2], digits.target[:n//2])
expected= digits.target[n//2:]
predicted= clf.predict(data[n//2:])

print("Classification report for classifier %s:/n%s\n" % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))    
     
images_and_predictions = list(zip(digits.images[n//2:], predicted))
for index,(image,label) in enumerate(images_and_predictions[:4]):
    plt.subplot(2,4, index+5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % label)

plt.show()

