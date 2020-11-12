"""
гиперболоид задаётся формулой -x1^2 + x2^2 + ... + xn^2 = -1
"""
from math import sinh, cosh, sqrt
import numpy as np


def scalar_product(v1: np.array, v2: np.array) -> float:
    """
    ищет скалярное произведение векторов в R(n+1)
    Необходимо передавать массивы одинаковой размерности.
    """
    assert len(v1) == len(v2)
    dimension = len(v1)
    ans = (v1[0] - v2[0]) ** 2
    for i in range(1, dimension):
        ans += (v2[i] - v1[i]) ** 2
    return ans


def norm(v: np.array) -> float:
    """
    ищет норму вектора в R(n+1)
    """
    return sqrt(scalar_product(v, np.zeros(len(v))))


def distance_pseudo_euclidean(v1: np.array, v2: np.array) -> float:
    """
    ищет расстояние между точкам в R(n+1)
    """
    return sqrt(scalar_product(v1, v2))


def hyperbolic_distance(p1: np.array, p2: np.array) -> float:
    """
    ищет расстояние между двумя точками в H(n)
    Необходимо передавать массивы одинаковой размерности
    """
    return np.arccosh(scalar_product(p1, p2))


def exponential_map(start_point: np.array, v: np.array, t: float) -> np.array:
    """
    вычисляет координаты точки. Стартовая точка - start_point, вектора направления в старотовой точке - v
    t - параметр, от которого зависит лишь расстояние между точками
    """
    nv = norm(v)
    ans1 = np.array([cosh(nv * t) * p for p in start_point])
    ans2 = np.array([sinh(nv * t) / nv * vi for vi in v])
    return ans1 + ans2


def rand_vector(point: np.array) -> np.array:
    """
    это функция должна возвращать случайный вектор,
    который находится в касательном подпространстве в точке point_coordinates[point].
    """
    dimension = len(point) - 1
    ans = np.zeros(dimension + 1)
    for i in range(1, dimension + 1):
        ans[i] = np.random.uniform(point[i] - 1, point[i] + 1)
    x0 = 0
    for i in range(1, dimension + 1):
        x0 += point[i] * (ans[i] - point[i])
    ans[0] = x0 / point[0]
    return ans
