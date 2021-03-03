import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
image = Image.open('./datasets/landscape.jpg')
origin_image = np.array(image)
image = origin_image.reshape((-1, 3))
shape = origin_image.shape
bandwidth = estimate_bandwidth(image, quantile = .2, n_samples = 100)
print(bandwidth)
clf = MeanShift(bandwidth = bandwidth, bin_seeding = True)
clf.fit(image)
labels = clf.labels_
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(origin_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(shape[:2]))
plt.axis('off')
plt.show()
