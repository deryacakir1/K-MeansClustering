# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:49:49 2023

@author: Derya
"""

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

import mglearn

"""
mglearn.plots.plot_kmeans_algorithm()

mglearn.plots.plot_kmeans_boundaries()



from sklearn.datasets import make_blobs

X,y_gercek=make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(X[:,0],X[:,1], s=50)


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)

kmeans.fit(X)

y_kmeans=kmeans.predict(X)
plt.scatter(X[:,0],X[:,1], s=50, c=y_kmeans, cmap='viridis')
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)



from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)
labels=KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1], s=50, c=labels, cmap='viridis')

from sklearn.cluster import SpectralClustering
model=SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels=model.fit_predict(X)
plt.scatter(X[:,0],X[:,1], s=50, c=labels, cmap='viridis')

"""

from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)

veri = china / 255
veri = veri.reshape(427 * 640, 3)

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(veri)

new_colors = kmeans.cluster_centers_
china_recolored = kmeans.predict(veri)
china_recolored = new_colors[china_recolored]
china_recolored = china_recolored.reshape(427, 640, 3)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)

plt.show()











