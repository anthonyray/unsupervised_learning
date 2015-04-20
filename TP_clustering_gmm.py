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

"""
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
XX, YY = np.meshgrid(x, y)
Z = np.log(-clf.score_samples(np.c_[XX.ravel(), YY.ravel()])[0]).reshape(XX.shape)
plt.close('all')
CS = plt.contour(XX, YY, Z)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'og')
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
plt.axis('tight')
plt.show()

"""

# Estimation du nombre de classes optimal par cross validation
# Constitution du jeu d'apprentissage et de test
X_train = X[::2]
X_test = X[1::2]

from sklearn import cross_validation
from sklearn.cross_validation import KFold

# Mise en place de la cross validation par K-fold
cross_val_scores = list()
for nb_components in range(1,20):
          scores = list()
          for train,test in KFold(X_train.shape[0],10):
                    clf = mixture.GMM(n_components=nb_components,covariance_type='full')
                    clf.fit(X_train)
                    scores.append(clf.score(X_test).sum())
          print 'For ' + str(nb_components) + ' classes, the average score is : ' + str(np.array(scores).mean())
          cross_val_scores.append(np.array(scores).mean())

plt.plot(np.arange(1,20),np.array(cross_val_scores))
plt.xlabel('Number of classes')
plt.ylabel('Score')
plt.show()

"""
Akaike information criterion & Bayesian information criterion
"""

aic_scores = list()
bic_scores = list()
for nb_components in range(1,20):
          clf = mixture.GMM(n_components=nb_components,covariance_type='full')
          clf.fit(X_train)
          aic_scores.append(clf.aic(X_train))
          bic_scores.append(clf.bic(X_train))


fig, axes = plt.subplots(3, 1, figsize=(10,4))
axes[0].plot(np.arange(1,20),np.array(cross_val_scores))
axes[0].set_xlabel('Number of classes')
axes[0].set_ylabel('Score')


axes[1].plot(np.arange(1,20),np.array(aic_scores))
axes[1].set_xlabel('Number of classes')
axes[1].set_ylabel('AIC')


axes[2].plot(np.arange(1,20),np.array(bic_scores))
axes[2].set_xlabel('Number of classes')
axes[2].set_ylabel('BIC')

plt.show()
