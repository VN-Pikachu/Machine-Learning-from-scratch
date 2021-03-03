import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;sns.set()
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.datasets import load_digits
from PIL import Image
data = load_digits()
model = KMeans(n_clusters = 10)
model.fit(data.data)
digits = model.cluster_centers_
fig, axes = plt.subplots(2,5)
for ax, num in zip(axes.flat, digits):
    ax.set(xticks = [], yticks = [])
    ax.imshow(num.reshape((8, 8)), cmap = 'binary')
plt.show()

image = np.array(Image.open('./datasets/landscape.jpg'))
origin_shape = image.shape
origin_image = image
image = image.reshape((-1, 3))
clf = KMeans(n_clusters = 3)
labels = clf.fit_predict(image)
plt.subplot(1, 2, 1)
plt.imshow(origin_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(origin_shape[:2]))
plt.axis('off')
plt.show()
