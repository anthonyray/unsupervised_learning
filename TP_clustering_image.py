# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:26:18 2014

@author: salmon
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


matrix_image=mpimg.imread('Grey_square_optical_illusion.png')
imgplot = plt.imshow(matrix_image)


f, axarr = plt.subplots(3,1)
axarr[0].imshow(matrix_image[:,:,0],cmap = plt.get_cmap('gray'))
axarr[0].set_title('Rouge')
axarr[1].imshow(matrix_image[:,:,1],cmap = plt.get_cmap('gray'))
axarr[1].set_title('Vert')
axarr[2].imshow(matrix_image[:,:,2],cmap = plt.get_cmap('gray'))
axarr[2].set_title('Bleu')

from scipy.ndimage import imread
from sklearn.cluster import KMeans


img = imread('Grey_square_optical_illusion.png',flatten=True)
orig_shape = img.shape
km = KMeans(n_clusters=2)
img = img.reshape((-1,1))
km.fit(img)
values = km.cluster_centers_.squeeze()
labels = km.labels_
# create an array from labels and values
img_compressed_2 = np.choose(labels, values)
img_compressed_2 = img_compressed_2.reshape(orig_shape)


img = imread('Grey_square_optical_illusion.png',flatten=True)
orig_shape = img.shape
km = KMeans(n_clusters=10)
img = img.reshape((-1,1))
km.fit(img)
values = km.cluster_centers_.squeeze()
labels = km.labels_
# create an array from labels and values
img_compressed_10 = np.choose(labels, values)
img_compressed_10 = img_compressed_10.reshape(orig_shape)

img = imread('Grey_square_optical_illusion.png',flatten=True)
orig_shape = img.shape
km = KMeans(n_clusters=10)
img = img.reshape((-1,1))
km.fit(img)
values = km.cluster_centers_.squeeze()
labels = km.labels_
# create an array from labels and values
img_compressed_25 = np.choose(labels, values)
img_compressed_25 = img_compressed_25.reshape(orig_shape)


imgplot = plt.imshow(img_compressed_2,cmap=plt.cm.gray, vmin=0, vmax=255)
imgplot = plt.imshow(img_compressed_10,cmap=plt.cm.gray, vmin=0, vmax=255)
imgplot = plt.imshow(img_compressed_25,cmap=plt.cm.gray, vmin=0, vmax=255)

