import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import math
data = pd.read_csv('./datasets/Pokemon.csv')
from sklearn.datasets import load_iris
iris = load_iris()
'''Visualize Each feature
fig, axes = plt.subplots(2,2,figsize = (10, 5))
fig.suptitle('Iris Feature Distribution')
for i, ax in enumerate(axes.flat):
    bins = np.linspace(math.floor(min(iris.data[:, i])), math.ceil(max(iris.data[:, i])), 25)
    for j, color in zip(range(3), ['red', 'green', 'blue']):
        ax.hist(iris.data[iris.target == j,i], bins = bins, color = color, alpha = .6,
        label = iris.target_names[j])
        ax.tick_params(axis = 'both', top = 'off', bottom = 'off', left = 'off', right = 'off',
        labelleft = 'on', labelbottom = 'on')
    for position in ['top', 'left', 'bottom', 'right']:
        ax.spines[position].set_visible(False)
    leg = ax.legend(loc = 1, fancybox = True, fontsize = 8)
    leg.get_frame().set_alpha(.5)
    ax.set_xlabel(iris.feature_names[i])

axes[0,0].set_ylabel('count')
axes[1,0].set_ylabel('count')

fig.tight_layout()
plt.show()
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
'''Uncomment to see the result of applying standardization and not applyiing standardization
#The overall result is the same
from sklearn.preprocessing import StandardScaler
iris.data = StandardScaler().fit_transform(iris.data)'''
model = LDA(n_components = 2)
X = model.fit_transform(iris.data, iris.target)
plt.scatter(X[:,0], X[:,1], c = iris.target, cmap = 'rainbow')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.title('Projected Samples')
plt.show()
