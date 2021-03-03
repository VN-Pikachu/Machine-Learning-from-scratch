from SparsityAutoencoder import SAE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
data, labels = digits.data, digits.target
data /= 16
''' when you change the number of units in hidden layer
    The cost function after being optimized is about the same for whatever number of units in hidden layer
    but the results are very different
    because as you increase the number of units in hidden layer
    the result will be better
    e.g: - when hidden_layer_size = 2, the final cost function is about 4.5
         but it learn very little
         (the decode images of different numbers are about the same)
         it learn some features with just 2 units (e.g: horizontal bar, vertical bar)
         - when hidden_layer_size = 16, the decode images is much more better
         numbers start forming shapes that are similar to the origin
         but the background is quite dark( noisy)
         - when hidden_Layer_size = 32, it can whiten the background'''


model = SAE([64,64,64], batch_size = 30, max_iter = 50, activation = 'identity', learning_rate = 1e-3, beta = .025, sparsity = .05)
model.fit(data, data)

plt.show()
X1 = data[:32]
X2 = model.decode(model.encode(X1))
fig, axes = plt.subplots(8,8, figsize = (15,10), subplot_kw = {'xticks': [], 'yticks':[]})
axes = axes.flat
def image(z): return z.reshape((8,8))
for a, b in zip(X1, X2):
    ax1 = next(axes)
    ax2 = next(axes)
    ax1.imshow(image(a) , cmap = 'binary')
    ax2.imshow(image(b), cmap = 'binary')

plt.tight_layout()
plt.grid()
plt.show()
