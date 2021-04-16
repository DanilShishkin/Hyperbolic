#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def hyper_scalar(point_1: np.array, point_2: np.array) -> float:
    ans = point_1*point_2
    ans[0] = -ans[0]
    return ans.sum()


def gradient(point_fix: np.array, point_move: np.array, distance: float) -> np.array:
    ans = 2. * point_fix * (np.arccosh(-hyper_scalar(point_fix, point_move)) -
                            distance)/np.sqrt(hyper_scalar(point_fix, point_move)**2 - 1.)
    return ans


def proj(current_point: np.array, grad: np.array) -> np.array:
    return grad + hyper_scalar(current_point, grad) * current_point


def exponential_map(x: np.array, v: np.array):
    norm_v = np.sqrt(hyper_scalar(v, v))
    return np.cosh(norm_v) * x + np.sinh(norm_v) * v / norm_v


def MSE(points, distance):
    n = distance.shape[0]
    error = 0.
    for i in range(n - 1):
        for j in range(i + 1, n):
            error += (np.arccosh(-hyper_scalar(points[i],
                                               points[j])) - distance[i, j])**2
    return error / n


def GD(points: np.array, distance: np.array, max_iter: int = 5000,
       start_rate=1e-3, lsize: float = 1.):

    for s in range(max_iter):
        sample_size = int(len(points) * lsize)
        grads = np.zeros(points.shape)

        points_indexes = np.arange(len(points))
        sample_ind = np.random.choice(  # индексы точек, по которым считаем градиент
            points_indexes, size=sample_size, replace=False)

        for i in sample_ind:
            for j in sample_ind:
                if distance[i, j] != 0.:
                    grad = gradient(points[i], points[j], distance[i][j])
                    grads[j] -= grad

        for k in sample_ind:
            pr = proj(points[k], grads[k])
            points[k] = exponential_map(points[k], -start_rate * pr)
            points[k, 0] = np.sqrt(1+sum(points[k, 1::]**2))
    return points
