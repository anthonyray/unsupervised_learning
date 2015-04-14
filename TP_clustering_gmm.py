import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

n_samples = 300
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          np.random.randn(n_samples, 2) + np.array([20, 20])]

clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X)
labels = clf.predict(X)
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
XX, YY = np.meshgrid(x, y)
Z = np.log(-clf.eval(np.c_[XX.ravel(), YY.ravel()])[0]).reshape(XX.shape)
plt.close('all')
CS = plt.contour(XX, YY, Z)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'og')
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
plt.axis('tight')
plt.show()

# Estimation du nombre de classes optimal par cross validation
X_train = X[::2]
X_test = X[1::2]
# Constitution du jeu d'apprentissage et de test
from sklearn import cross_validation
clf = mixture.GMM(n_components=2, covariance_type='full')
cross_validation.cross_val_score(clf,X,cv=2)
clf.fit(X_train)
clf.score(X_train)