import pandas as pd
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import axes as ax
import numpy as np


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() -1, x.max() +1
    y_min, y_max = y.min() -1, y.max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# method used to display how degree of polynomial effects separator
def deg_test():
    data_file = 'Iris_Data_Set/iris.data'
    iris_data = pd.read_csv(data_file)
    iris_data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
    np_data = iris_data.to_numpy()
    X = np_data[:, :2]
    y = []
    for name in np_data[:, 4]:
        switch = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2,
        }
        y.append(switch.get(name, -1))
    C = 1.0  # SVM regularization parameter
    deg1 = 1
    deg2 = 2
    deg3 = 3
    deg4 = 4
    # testing degree 1, 2, 3, and 4 polynomials
    models = (svm.SVC(kernel='poly', degree=deg1, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=deg2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=deg3, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=deg4, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with polynomial (degree ' + str(deg1) + ') kernel',
              'SVC with polynomial (degree ' + str(deg2) + ') kernel',
              'SVC with polynomial (degree ' + str(deg3) + ') kernel',
              'SVC with polynomial (degree ' + str(deg4) + ') kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=1, hspace=0.7)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        #ax.set_xticks(())
        #ax.set_yticks(())
        ax.set_title(title)

    plt.show()


# example code copied from scikit website
def online():
    data_file = 'Iris_Data_Set/iris.data'
    iris_data = pd.read_csv(data_file)
    iris_data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
    np_data = iris_data.to_numpy()
    X = np_data[:, :2]
    y = []
    for name in np_data[:, 4]:
        switch = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2,
        }
        y.append(switch.get(name, -1))
    C = 1.0  # SVM regularization parameter
    deg = 3
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=10000),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=deg, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree '+str(deg)+') kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


# practicing linear support vector machine using iris data set imported with pandas
# using scikit-learn library
def linearSVM():
    data_file = 'Iris_Data_Set/iris.data'
    iris_data = pd.read_csv(data_file)
    iris_data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
    np_data = iris_data.to_numpy()
    X = np_data[:, :2]
    y = []
    for name in np_data[:, 4]:
        switch = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2,
        }
        y.append(switch.get(name, -1))
    linearsvc_model = svm.LinearSVC(penalty='l1', dual=False, max_iter=10000)
    linearsvc_model.fit(X, y)
    clf = svm.SVC()
    clf.fit(X,y)
    print(clf.support_vectors_)
    sub = plt.subplot()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0,X1)
    plot_contours(sub, linearsvc_model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    sub.set_xlim(xx.min(), xx.max())
    sub.set_ylim(yy.min(), yy.max())
    sub.set_xlabel('Sepal length')
    sub.set_ylabel('Sepal width')
    sub.set_xticks(())
    sub.set_yticks(())
    sub.set_title('Linear SVM')
    plt.show()


linearSVM()
online()
deg_test()
