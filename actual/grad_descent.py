#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def hyper_scalar(point_1: np.array, point_2: np.array) -> float:
    ans = point_1*point_2
    ans[0] = -ans[0]
    return ans.sum()


def grad_euclid_g(point_fix: np.array, point_move: np.array, distance: float) -> np.array:
    ans = 2. * point_fix * (np.arccosh(-hyper_scalar(point_fix, point_move)) -
                            distance)/np.sqrt(hyper_scalar(point_fix, point_move)**2 - 1.)
    return ans


def proj(current_point: np.array, grad: np.array) -> np.array:
    return grad + hyper_scalar(current_point, grad) * current_point


def expon(x: np.array, v: np.array):
    norm_v = np.sqrt(hyper_scalar(v, v))
    return np.cosh(norm_v) * x + np.sinh(norm_v) * v / norm_v


def GD(points: np.array, distance: np.array, max_iter: int = 5000, start_rate=1e-3):
    for s in range(max_iter):
        grads = np.zeros(points.shape)
        err = 0
        for i in range(len(points)):
            for j in range(len(points)):
                if distance[i, j] != 0:
                    grad = grad_euclid_g(points[i], points[j], distance[i][j])
                    grads[j] -= grad
        for k in range(len(points)):
            pr = proj(points[k], grads[k])
            points[k] = expon(points[k], -start_rate * pr)
            points[k, 0] = np.sqrt(1+sum(points[k, 1::]**2))
        for p in range(len(points)):
            for o in range(len(points)):
                if distance[p, o] != 0:
                    err += (np.arccosh(-hyper_scalar(
                        points[p], points[o]))-distance[p, o])**2
        # print(s, ':', err)
    return points


def rand_point(dimension: int, max: float) -> np.array:
    ans = np.zeros(dimension + 1)
    ssum = 0
    for i in range(1, dimension + 1):
        ans[i] = max*np.random.uniform() - max/2
        ssum += ans[i]**2
    ans[0] = np.sqrt(ssum+1)
    return ans


def MSE(points, distance):
    n = distance.shape[0]
    error = 0.
    for i in range(n - 1):
        for j in range(i + 1, n):
            error += (np.arccosh(-hyper_scalar(points[i],
                                               points[j])) - distance[i, j])**2
    return error / n
