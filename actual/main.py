"""
главная функци с классом Hyperbolic
"""
import Hyperbolic
import numpy as np
import Levenshtein as Lev
import pandas as pd
from grad_descent import MSE


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

size = 100
dataset = np.concatenate((positive[np.random.choice(
    len(positive), size)], negative[np.random.choice(len(negative), size)]))

perm = np.random.permutation(2*size)
ran = np.array(range(2*size))
map = {perm[i]: ran[i] for i in range(2*size)}
dataset = dataset[perm]

distance = np.zeros((2*size, 2*size), dtype=np.int32)

for i in range(2*size):
    for j in range(2*size):
        distance[i, j] = Lev.distance(dataset[i, 0], dataset[j, 0])

distance = distance / (3*size)

H = Hyperbolic.Hyperbolic(graph=distance, dimension=2, maxiter=100, batch=0.2)
print(MSE(H.point_coordinates, distance))

H.draw(False, map=map)
