from LDA import *
import seaborn as sns;sns.set()
samples = np.array([
[1,2],
[2,3],
[3,3],
[4,5],
[5,5],
[4,2],
[5,0],
[5,2],
[3,2],
[5,3],
[6,3]
]).astype('float64')
labels = np.array([0] * 5 + [1] * 6)
def Gaussian_Kernel(t):
    return np.exp(-t/2)
def plot_KDE(X, c, h = .5):
    x = np.linspace(min(X) - 1, max(X) + 1, 50)
    y = [np.sum(Gaussian_Kernel(np.linalg.norm(val - X, axis = 1) ** 2 / h ** 2)) / len(X) for val in x]
    plt.plot(x, y , color = c, linewidth = 1)
plt.subplot(2,1, 1)
plt.scatter(samples[:, 0], samples[:, 1], c = labels, cmap = 'rainbow')

model = LDA(n_components = 1)
model.fit(samples, labels)
x_new = model.transform(samples)

plt.subplot(2,1,2)
plt.scatter(x_new[:,0], np.zeros(len(x_new)), c = labels, cmap = 'rainbow')
plot_KDE(x_new[labels == 0], 'blue', .35)
plot_KDE(x_new[labels == 1], 'red', .35)
plt.xlabel('projected data')
plt.ylabel('P(projected data)')
plt.axis('tight')
plt.legend(['Class 1', 'Class 2'])
plt.show()
