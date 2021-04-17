#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from hyperbolic import scalar_product


def gradient(point_fix: np.array, point_move: np.array, distance: float) -> np.array:
    """
    Градиент функции arccosh(-(x, y)_L), 
    где (x, y)_L = -x_0 * y_0 + sum_{i=1}^{n} x_i * y_i
    в точках point_fix, point_move

    Параметры:
    ___________

    point_fix : np.array
        Фиксированная точка

    point_move : np.array
        Точка, которая двигается в градиетном спуске

    distance : float 
        Изначальное расстояние между point_fix, point_move

    Возвращает:
    ____________

    grad : np.array
        Значение градиента в точках point_fix, point_move
    """
    grad = 2. * point_fix * (np.arccosh(-scalar_product(point_fix, point_move)) -
                             distance)/np.sqrt(scalar_product(point_fix, point_move)**2 - 1.)
    return grad


def proj(current_point: np.array, grad: np.array) -> np.array:
    """
    Ортогональная проекция grad из R^(n+1) 
    в касательное пространство к точке current_point

    Параметры: 
    ___________

    current_point : np.array
        Точка, на чье касательное пространство проецируется grad
    grad : np.array
        Точка, которая проецируется

    Возвращает:
    ___________
    np.array
        Точка в касательном пространстве к current_point
    """
    return grad + scalar_product(current_point, grad) * current_point


def exponential_map(x: np.array, v: np.array):
    """
    Проекция из касательного пространства к x.

    Параметры:
    __________

    x : np.array
        Точка, из чьего касательного пространства проецируется v

    v : np.array
        Точка, которая проецируется на H(n)

    Возвращает:
    ____________
    np.array
    Координаты точки v на H(n)
    """
    norm_v = np.sqrt(scalar_product(v, v))
    return np.cosh(norm_v) * x + np.sinh(norm_v) * v / norm_v


def MSE(points: np.array, distance: np.array):
    """
    Среднеквадратичное отклонение расстояний 
    между точками на гиперболоиде от изначальных расстояний

    Параметры:
    ___________

    points : np.array, 2-dimensional
        Координаты точек на гиперболоиде

    distance : np.array, 2-dimensional
        Матрица смежности графа

    Возвращает:
    ___________

    error : float
        Среднеквадратичная ошибка

    """
    n = distance.shape[0]
    error = 0.
    for i in range(n - 1):
        for j in range(i + 1, n):
            error += (np.arccosh(-scalar_product(points[i],
                                                 points[j])) - distance[i, j])**2
    return error / n


def GD(points: np.array, distance: np.array, max_iter: int = 5000,
       start_rate=1e-3, batch: float = 1.):
    """
    Стохастический градиентный спуск. Минимизирует среднеквадратичное 
    отклонение расстояний между точками на гиперболоиде от исходных расстояний.

    Параметры:
    ___________
    points : np.array, 2-dimensional array
        Координаты точек на гиперболоиде

    distance : np.array
        Исходные расстояния между точками

    max_iter : int
        Количество инераций градиентного спуска

    start_rate : float
        Шаг градиентного спуска

    batch : float
        Процент точек, от которых считается градиент на каждом шаге.
        Может быть не больше 1. и не меньше 0.

    Возвращает:
    ___________

    points : np.array
        Новые координаты точек на гиперболоиде с минимальным MSE
    """
    for s in range(max_iter):

        sample_size = int(len(points) * batch)
        grads = np.zeros(points.shape)

        points_indexes = np.arange(len(points))

        sample_ind = np.random.choice(  # индексы точек, по которым считаем градиент
            points_indexes, size=sample_size, replace=False)

        for i in sample_ind:
            for j in sample_ind:
                if distance[i, j] != 0.:
                    # Подсчет градиетна
                    grad = gradient(points[i], points[j], distance[i][j])
                    grads[j] -= grad

        for k in sample_ind:
            # проекция градиента на гиперболоид, шаг в сторону антиградиента
            pr = proj(points[k], grads[k])
            points[k] = exponential_map(points[k], -start_rate * pr)
            points[k, 0] = np.sqrt(1+sum(points[k, 1::]**2))
    return points
