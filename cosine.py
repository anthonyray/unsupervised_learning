# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:31:50 2015

@author: anthonyreinette
"""
from time import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from TP_clustering_kmeans_cosine import *

categories = ['alt.atheism','comp.graphics']

print "Loading 20 newsgroups dataset for categories:"
print categories

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             data_home='../trash',
                             shuffle=True, random_state=42)

print "%d documents" % len(dataset.data)
print "%d categories" % len(dataset.target_names)

labels = dataset.target
true_k = np.unique(labels).shape[0]
print "Extracting features from the training set using a sparse vectorizer"
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             stop_words='english')
X = vectorizer.fit_transform(dataset.data)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape

X = X.toarray()

centroids, labels = kmeans(X,true_k)