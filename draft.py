import numpy as np
import matplotlib.pyplot as plt
def graph(y,n, states, ax):
    y = np.array(y)
    states = np.array(states)
    ax.plot(np.arange(len(y)), y)
    MIN = np.argmin(y)
    ax.scatter(MIN, y[MIN], c= 'r', s=50, marker = '^')
    ax.scatter(states, y[states], c = 'g', marker = 'x')
    states = (states + n //2) % n
    ax.scatter(states, y[states], c = 'y', marker = 'd')
    ax.legend(['Min:%d' % MIN])
def f(state, k):
    return [sum(min(abs(x-val), k - abs(x-val)) for x in state) for val in range(0, k)]
def plot(states, ks):
    n = len(states)
    fig, axes = plt.subplots(n, 1, figsize = (20, 40),gridspec_kw = {'hspace':.5})
    Y = [f(s, k) for s, k in zip(states, ks)]
    for y, n, s, ax in zip(Y, ks, states, axes.flat):
        graph(y,n,s,ax)
    plt.show()
states = [[2, 7, 1], [2, 0, 1, 2, 0, 1, 2],[1, 3],[7, 8, 9, 3, 3],[97, 98, 99, 0, 1],
[178, 104, 21, 81, 330, 353, 299, 263, 221, 199, 124, 261, 66, 204, 244, 337, 224, 84, 352, 91],
[45, 103, 44, 107, 41, 182, 14, 53, 181, 140, 186, 271, 189, 110, 78, 208, 354, 350, 70, 231]]
labels = [10,3,4,10,100,360,360]
plot(states, labels)
