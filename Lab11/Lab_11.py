from sklearn import datasets
breast_cancer = datasets.load_breast_cancer()
breast_data = breast_cancer.data
breast_labels = breast_cancer.target

print(breast_data.shape)
print(breast_labels.shape)

import numpy as np
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape

import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)
features = breast_cancer.feature_names
features

final_breast_data[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_data,
        breast_labels, random_state=26)

print(X_train.shape, X_test.shape)

"""Preprocessing: Principal Component Analysis
-------------------------------------------

We can use PCA to reduce these features to a manageable size, while maintaining most of the information
in the dataset.


"""

from sklearn import decomposition
pca = decomposition.PCA(n_components=20, whiten=True)
pca.fit(X_train)

"""The principal components measure deviations about this mean along
orthogonal axes.


"""

print(pca.components_.shape)

"""With this projection computed, we can now project our original training
and test data onto the PCA basis:


"""

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

print(X_test_pca.shape)

"""
Doing the Learning: Support Vector Machines
-------------------------------------------

Now we'll perform support-vector-machine classification on this reduced
dataset:

"""

from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

from sklearn import metrics
y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))

"""Another interesting metric is the *confusion matrix*, which indicates
how often any two items are mixed-up. The confusion matrix of a perfect
classifier would only have nonzero entries on the diagonal, with zeros
on the off-diagonal:


"""

print(metrics.confusion_matrix(y_test, y_pred))

"""With Iris Dataset"""

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

print(iris_data.shape)
print(iris_labels.shape)

features = iris.feature_names
features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_data,
        iris_labels, random_state=26)

print(X_train.shape, X_test.shape)

"""Preprocessing: Principal Component Analysis

We can use PCA to reduce these features to a manageable size, while maintaining most of the information in the dataset.
"""

from sklearn import decomposition
pca = decomposition.PCA(n_components=2, whiten=True)
pca.fit(X_train)

print(pca.components_.shape)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

print(X_test_pca.shape)

from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

from sklearn import metrics
y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))
