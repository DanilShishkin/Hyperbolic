"""
главная функци с классом Hyperbolic
"""
import Hyperbolic
import numpy as np
import Levenshtein as Lev
import pandas as pd
from grad_descent import MSE
import draw


def is_on_hyperbola(point):
    return -point[0]**2 + point[1]**2 + point[2]**2 == -1.


positive = np.array(pd.read_csv(
    r'positive.csv', sep=';', usecols=[3], names=['text']))
negative = np.array(pd.read_csv(
    r'negative.csv', sep=';', usecols=[3], names=['text']))

# positive_90 = np.array(
#     positive[positive['text'].apply(lambda text: len(text) == 90)])
# negative_90 = np.array(
#     negative[negative['text'].apply(lambda text: len(text) == 90)])

# size = 100
# dataset = np.concatenate((positive[np.random.choice(
#     len(positive), size)], negative[np.random.choice(len(negative), size)]))

# perm = np.random.permutation(2*size)
# ran = np.array(range(2*size))
# map = {perm[i]: ran[i] for i in range(2*size)}

# dataset = dataset[perm]

# distance = np.zeros((2*size, 2*size), dtype=np.int32)

# for i in range(2*size):
#     for j in range(2*size):
#         distance[i, j] = Lev.distance(dataset[i, 0], dataset[j, 0])

# distance = distance / 5.

# print(distance.max())

# H = Hyperbolic.Hyperbolic(graph=distance, dimension=2, maxiter=100, batch=0.1)
# print(MSE(H.point_coordinates, distance))

# draw.draw(H.point_coordinates, distance,
#           draw_edges=False, map=map, annotate=False)

lst = np.array([]).reshape(0, 0)
for i in range(100):
    f = open(rf'data/sequence ({i}).txt', 'r')
    string = f.read().replace("\n", "")
    tmp = list(string.encode())
    if i == 0:
        lst = np.array(tmp).reshape(1, -1)
    else:
        lst = np.vstack((lst, np.array(tmp).reshape(1, -1)))
# создал пустой интовый массив для расстояний
distance = np.zeros((100, 100), dtype=float)
# заполнил его правильными значениями
for i in range(100):
    for j in range(100):
        distance[i, j] = (lst[i] != lst[j]).sum()

distance = distance / 2.
H = Hyperbolic.Hyperbolic(graph=distance, dimension=2, maxiter=100, batch=0.1)
draw.draw(H.point_coordinates, distance, True, annotate=False)
