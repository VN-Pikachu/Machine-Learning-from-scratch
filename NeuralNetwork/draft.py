from NeuralNetwork import *
import numpy as np
from scipy import sparse

def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()
X = [["....................",   "........###.........",   ".....####.####......",   "....##.......##.....",   "...##.........##....",   "...#...........#....",   "...............##...",   "................#...",   "...............##...",   "...............#....",   "..............##....",   "............##......",   "...........##.......",   ".........##.........",   "..########..........",   "..#..#####.......##.",   "..####....########..",   "..##................",   "....................",   "...................."],
["....................",   "....................",   ".........#..........",   "........##..........",   ".......#............",   "......##............",   "......#.............",   ".....##.............",   "....##..............",   "....#...............",   "...##.....#.........",   "..##......#.........",   "..#.......#.........",   ".##############.....",   "..........#.........",   ".........##.........",   ".........#..........",   ".........#..........",   "....................",  "...................."],
["....................",   "....................",   "..............##....",   ".....#########......",   ".....#..............",   ".....#..............",   "....#...............",   "....#...............",   "....#...............",   "....########........",   "............##......",   "..............#.....",   "..............##....",   "...............#....",   "..............#.....",   "....#.......##......",   "....#####.##........",   "....................",   "....................",   "...................."],
["....................",   "....................",   "....................",   "...###############..",   "..##............##..",   "...............##...",   "...............#....",   "..............#.....",   ".............##.....",   ".............#......",   "............##......",   "............#.......",   "........#########...",   "........#..#........",   "..........##........",   "..........#.........",   ".........#..........",   "........##..........",   "........#...........",   ".......##..........."],
["....................",   "....................",   ".......####.........",   ".....##...###.......",   "....#.......#.......",   "....#.......##......",   "....#........#......",   "....#........#......",   "....#......##.......",   ".....########.......",   "............#.......",   "...........##.......",   "...........#........",   "..........##........",   "..........#.........",   ".........##.........",   ".........#..........",   "........##..........",   ".......##...........",   "...................."],
["....................",   "....................",   "..........#########.",   ".....#####..........",   "....##..............",   "...##...............",   "...#................",   "..##................",   "..#.................",   ".########...........",   ".#.......####.......",   ".............##.....",   "..............##....",   "...............#....",   "...............#....",   "...............#....",   "...#..........##....",   "...###.......##.....",   ".....#########......",   "...................."],
["....................",   "....................",   "........###.........",   "......###.####......",   ".....##......#......",   ".....#.......#......",   "....##.......#......",   "....#........#......",   "....##......#.......",   ".....#......#.......",   "......######........",   "..........##........",   ".......###..##......",   "......#.......#.....",   ".....#.........#....",   ".....#.........#....",   "....#..........#....",   "....##........##....",   "......###....#......",   "..........###......."],
["....................",   "......#####.........",   ".....#....##........",   "...........##.......",   "............#.......",   "............#.......",   "............#.......",   "...........##.......",   ".........##.........",   "........##..........",   "..........##........",   "...........#........",   "...........##.......",   "............#.......",   "..#.........#.......",   "..#........##.......",   "..##.......#........",   "...##....##.........",   "....#####...........",   "...................."],
["....................",   "....................",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   "........#...........",   "........#...........",   "........#...........",   ".......#............",   ".......#............",   ".......#............",   ".......#............",   ".......#............",   "......#.............",   "......#.............",   "......#.............",   "...................."],
["....................",   ".........#..........",   ".........#..........",   "........#...........",   ".......##...........",   ".......#............",   "......##............",   "......#.............",   ".....##.............",   ".....#..............",   "....##......#.......",   "...##.......#.......",   "...#........#.......",   "...#........#.......",   "..###############...",   "............#.......",   "............#.......",   "............#.......",   "............#.......",   "...................."],
["....................",   "....................",   ".......############.",   ".......#............",   "......##............",   "......#.............",   "......#.............",   "......#.............",   "......######........",   "............##......",   ".............##.....",   "..............#.....",   "..............##....",   "...............#....",   "..............#.....",   "...##.........#.....",   ".....###..####......",   ".......####.........",   "....................",   "...................."],
["....................",   "....................",   "......######........",   ".....#.....####.....",   ".....#........##....",   ".....#.........#....",   "....##..........#...",   "....#...........#...",   "....#...........#...",   "....#...........#...",   "....#...........#...",   "....#...........#...",   "....#...........#...",   "....#...........#...",   "....#..........##...",   "....##.........#....",   ".....##.......##....",   "......###....#......",   "........#####.......",   "...................."],
["....................",   "....................",   "........#...........",   ".......#............",   "......##............",   "......#.............",   ".....#..............",   "....##..............",   "....#........#......",   "...##........#......",   "...#.........#......",   "...#.........#......",   "..##.........#......",   ".##..........#......",   "..#########.##..#...",   "..........#######...",   "............#.......",   "............#.......",   "............##......",   ".............#......"],
["....................",   "....................",   ".....#######........",   "....#......##.......",   "....#.......#.......",   "....#.......#.......",   "....#......##.......",   "....#.....###.......",   "....######..#.......",   "............#.......",   "............#.......",   "...........#........",   "...........#........",   "...........#........",   "...........#........",   "...........#........",   "...........#........",   "..........##........",   "....................",   "...................."],
["....................",   "....................",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   "........#...........",   "........#...........",   "........#...........",   "........#...........",   ".......##...........",   ".......#............",   ".......#............",   ".......#............",   ".......#............",   ".......#............",   "......##............",   "...................."],
["....................",   "....................",   ".....######.........",   ".....#....###.......",   "............#.......",   "............##......",   "............##......",   "............#.......",   "...........##.......",   "..........#.........",   ".........#####......",   ".............#......",   ".............#......",   ".............#......",   ".............#......",   ".............#......",   "...##.......#.......",   "....#######.........",   "....................",   "...................."],
["....................",   "....................",   ".......####.........",   ".....##....##.......",   "....#........##.....",   "..............##....",   "...............#....",   "...............#....",   "..............#.....",   ".............##.....",   ".............#......",   "............#.......",   "............#.......",   "...........#........",   ".........##.........",   ".........#..........",   ".......##...........",   ".....##.............",   ".....############...",   "...................."],
["....................",   "....................",   "..........####......",   ".......####..#......",   "....###.....#.......",   "...........##.......",   "..........##........",   "..........#.........",   ".........#..........",   "........#######.....",   "..............##....",   "...............#....",   "...............#....",   "..............##....",   ".............##.....",   "...........###......",   "......#####.........",   "....##..............",   "....................",   "...................."],
["....................",   "....................",   "..........##........",   "........###.........",   "........#...........",   ".......#............",   "......##............",   "......#.............",   "......#.............",   "......#.............",   "......#...###.......",   "......#.###.####....",   "......###......#....",   ".......#.......#....",   ".......##......#....",   "........########....",   "....................",   "....................",   "....................",   "...................."],
["....................",   "....................",   "...#############....",   "...#................",   "...#................",   "..##................",   "..#.................",   "..#.................",   ".##.................",   ".###############....",   "...............##...",   "................##..",   ".................#..",   ".................#..",   "................#...",   ".....#.........##...",   ".....###......##....",   ".......########.....",   "....................",   "...................."],
["....................",   "....................",   ".......#............",   ".......#............",   "........#...........",   "........#...........",   "........#...........",   "........#...........",   "........#...........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........##.........",   "..........#.........",   "..........#.........",   "..........#.........",   "...................."],
["....................",   "....................",   ".........#..........",   ".......###..........",   "......##.#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   ".........#..........",   "......#######.......",   "....................",   "...................."],
["....................",   "....................",   ".......##...........",   "......##.###........",   "...........##.......",   "............#.......",   "............#.......",   "............#.......",   "..........##........",   ".........###........",   "............##......",   ".............#......",   ".............##.....",   "..............#.....",   ".............##.....",   ".............#......",   "....#.......#.......",   "....########........",   "....................",   "...................."],
["....................",   "....................",   ".......####.........",   ".....###..###.......",   ".....#......##......",   "....##.......##.....",   "....#.........##....",   "...##..........##...",   "...#............#...",   "...#............#...",   "..#..............#..",   "..#..............#..",   "..#..............#..",   "..##.............#..",   "...#............#...",   "...##...........#...",   ".....#.........##...",   "......##.....##.....",   "........#####.......",   "...................."],
["....................",   "....................",   "....############....",   "..............##....",   ".............##.....",   "............##......",   "............#.......",   "............#.......",   "...........##.......",   "...........#........",   "...........#........",   "..........#.........",   "..........#.........",   "..........#.........",   ".........##.........",   ".........#..........",   "........##..........",   "........#...........",   "........#...........",   "...................."],
["....................",   "........###.........",   "......##...##.......",   ".....#.......#......",   "....#.........#.....",   "....#.........#.....",   "....#.........#.....",   ".....#.......#......",   "......#.....#.......",   ".......#####........",   ".....##.....##......",   "....#.........#.....",   "...#...........#....",   "...#...........#....",   "..#.............#...",   "..#.............#...",   "..#.............#...",   "...#...........#....",   "....##.......##.....",   "......#######......."],
["....................",   "...##############...",   "..#..............#..",   ".................#..",   "................#...",   "...............#....",   "..............#.....",   ".............#......",   "............#.......",   "............#.......",   ".......########.....",   "...........#........",   "..........#.........",   "..........#.........",   ".........#..........",   ".........#..........",   "........#...........",   "........#...........",   "........#...........",   "........#..........."],
["................#...",   ".....###########....",   "....#...............",   "....#...............",   "...#................",   "...#................",   "..#.................",   "..#.................",   "..######............",   "........#####.......",   ".............##.....",   "...............#....",   "................#...",   "................#...",   "................#...",   "................#...",   "...............#....",   "...............#....",   "....#.........#.....",   ".....#########......"],
["....................",   "........###.........",   "......##...##.......",   ".....#.......#......",   "....#.........#.....",   "....#.........#.....",   "...#...........#....",   "...#...........#....",   "...#...........#....",   "..#.............#...",   "..#.............#...",   "..#.............#...",   "...#...........#....",   "...#...........#....",   "...#...........#....",   "....#.........#.....",   "....#.........#.....",   ".....#.......#......",   "......##...##.......",   "........###........."],
["....................",   ".........#..........",   ".........#..........",   "........#...........",   ".......#............",   ".......#............",   "......#.............",   ".....#..............",   ".....#......#.......",   "....#.......#.......",   "...#........#.......",   "...#........#.......",   "..##############....",   "............#.......",   "............#.......",   "............#.......",   "............#.......",   "............#.......",   "............#.......",   "...................."]]
y = [2, 4,5, 7, 9, 5, 8, 3, 1, 4, 5, 0, 4, 9, 1, 3, 2, 3, 6, 5, 1, 1, 3,  0, 7, 8, 7, 5, 0, 4]
X = np.array([np.array(list(''.join(img))) for img in X])
X[X == '.'] = 0
X[X == '#'] = 1
X = X.astype('float64')
'''from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
data, labels = digits.data, digits.target

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2, random_state = 6)
model.fit(x_train, one_hot_coding(y_train))
print(model.predict(X))'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import sparse
def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()

x_train, y_train = X, y
print(x_train.shape)
print(x_train)
print(y_train)
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
X = LDA(n_components = 2).fit_transform(X, y)
model = NeuralNetwork([2, 32, 10], max_iter = 20, batch_size = 20, learning_rate = 3, activation = 'logistic')
model.fit(x_train, one_hot_coding(y_train))
print(list(model.W[1:]))

print(list(model.b[1:]))