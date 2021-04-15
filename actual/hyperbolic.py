"""
гиперболоид задаётся формулой -x1^2 + x2^2 + ... + xn^2 = -1
"""
from math import sinh, cosh, sqrt
import numpy as np


def scalar_product(point_1: np.array, point_2: np.array) -> float:
    """
    ищет скалярное произведение векторов в R(n+1)
    Необходимо передавать массивы одинаковой размерности.
    """
    ans = point_1*point_2
    ans[0] = -ans[0]
    return ans.sum()


def norm(v: np.array) -> float:
    """
    ищет норму вектора в R(n+1)
    """
    return sqrt(scalar_product(v, v))


def distance_pseudo_euclidean(v1: np.array, v2: np.array) -> float:
    """
    фигня
    ищет расстояние между точкам в R(n+1)
    """
    return sqrt(scalar_product(v1, v2))


def hyperbolic_distance(p1: np.array, p2: np.array) -> float:
    """
    ищет расстояние между двумя точками в H(n)
    Необходимо передавать массивы одинаковой размерности
    """
    inner = scalar_product(p1, p2)
    if -inner < 1.:
        inner = -1.
    return np.arccosh(-inner)


def exponential_map(start_point: np.array, v: np.array, t: float) -> np.array:
    """
    вычисляет координаты точки. Стартовая точка - start_point, вектор направления 
    в старотовой точке - v
    t - параметр, от которого зависит лишь расстояние между точками
    """
    nv = np.linalg.norm(v)
    return np.cosh(t) * start_point + np.sinh(t) / nv * v


def rand_vector(point: np.array) -> np.array:
    """
    это функция должна возвращать случайный вектор,
    который находится в касательном подпространстве 
    к точке point_coordinates[point].
    """

    dimension = len(point)
    tang_vector = np.zeros(dimension)
    tang_vector[1:] = np.random.uniform(0., 1., size=dimension - 1)
    tang_vector[0] = sum(point[1:] * tang_vector[1:]) / point[0]
    return tang_vector


def projection(coordinates):
    return np.array([coordinates[i, 1:] / (coordinates[i, 0] + 1)
                     for i in range(coordinates.shape[0])])
