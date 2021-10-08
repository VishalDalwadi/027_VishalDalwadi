import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.datasets import mnist

(features_train, targets_train), (features_test, targets_test) = mnist.load_data()
# Convert to float32.

features_train, features_test = np.array(features_train, np.float32), np.array(features_test, np.float32)

# Flatten images to 1-D vector of 784 features (28*28).
num_features=784

features_train, features_test = features_train.reshape([-1, num_features]), features_test.reshape([-1, num_features])

# Normalize images value from [0, 255] to [0, 1].

features_train, features_test = features_train / 255., features_test / 255

print(len(features_train))
print(len(features_test))

# Create a linear SVM classifier
clf = svm.SVC(kernel='linear')

# Train classifier
clf.fit(features_train, targets_train)

# Make predictions on unseen test data
clf_predictions = clf.predict(features_test)

print("Accuracy: {}%".format(clf.score(features_test, targets_test) * 100 ))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(clf_predictions, targets_test))

# Create a polynomial SVM classifier
clf1 = svm.SVC(kernel='poly')

# Train classifier
clf1.fit(features_train, targets_train)

# Make predictions on unseen test data
clf1_predictions = clf1.predict(features_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(clf1_predictions, targets_test))

# Create a rbf SVM classifier
clf2 = svm.SVC(kernel='rbf')

# Train classifier
clf2.fit(features_train, targets_train)

# Make predictions on unseen test data
clf2_predictions = clf2.predict(features_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(clf2_predictions, targets_test))

"""From above result we can say that rbf is better than polinomial and linear
#RBF >= Poli > Linear

"""
