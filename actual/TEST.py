#!/usr/bin/python3
"""
тестировщик
"""
import main
import hyperbolic
import numpy as np


def points():
    """
    тестирует принадлежность точки гиперболоиду
    """
    NUM_OF_TESTS = 1
    MIN_NUM_OF_POINT = 100
    MAX_NUM_OF_POINTS = 100
    MIN_DIMENSION = 3
    MAX_DIMENSION = 3
    # FIXME СТРАННО, НО В КАКИХ-ТО СЛУЧАЯХ ЕСТЬ ОШИБКИ. НУЖНО ВЫЯСНИТЬ ПОЧЕМУ
    MAX_POINT_ERROR = 2
    MIN_DISTANCE = 1
    MAX_DISTANCE = 20
    flag = 0
    counter = 0
    for test in range(NUM_OF_TESTS):
        N = np.random.randint(
            MIN_NUM_OF_POINT, MAX_NUM_OF_POINTS + 1, size=1, dtype=int)
        N = N[0]
        a = MIN_DISTANCE + np.random.ranf([N, N]) * MAX_DISTANCE
        matrix = np.tril(a) + np.tril(a, -1).T
        for e in range(len(matrix[0])):
            matrix[e][e] = 0
        dimension = np.random.randint(
            MIN_DIMENSION, MAX_DIMENSION + 1, size=1, dtype=int)
        dimension = dimension[0]
        H = main.Hyperbolic(matrix, dimension)
        for point in H.point_coordinates:
            counter += 1
            ans = sum(map(lambda x: x ** 2, point))
            ans -= 2 * point[0] ** 2
            if abs(ans + 1) > MAX_POINT_ERROR:
                print("Point", point, "not in H+")
                flag += 1

    print(counter, "points checked", flag, "points are wrong")
    H.print_graph("blue")


def distance():
    """
    тестирует правильность расстояния меджу точками
    """
    NUM_OF_TESTS = 1
    MIN_NUM_OF_POINT = 100
    MAX_NUM_OF_POINTS = 100
    MIN_DIMENSION = 3
    MAX_DIMENSION = 3
    MAX_DISTANCE_ERROR = 10
    MIN_DISTANCE = 1
    MAX_DISTANCE = 20
    flag = 0
    counter = 0
    for test in range(NUM_OF_TESTS):
        N = np.random.randint(
            MIN_NUM_OF_POINT, MAX_NUM_OF_POINTS + 1, size=1, dtype=int)
        N = N[0]
        a = MIN_DISTANCE + np.random.ranf([N, N]) * MAX_DISTANCE
        matrix = np.tril(a) + np.tril(a, -1).T
        for e in range(1, len(matrix[0])):
            for k in range(1, len(matrix[0])):
                matrix[e][k] = 0
        matrix[0][0] = 0
        dimension = np.random.randint(
            MIN_DIMENSION, MAX_DIMENSION + 1, size=1, dtype=int)
        dimension = dimension[0]
        H = main.Hyperbolic(matrix, dimension)
        # H.print_graph()  # DEBUG это для
        for i in range(1, len(H.point_coordinates)):
            counter += 1
            d = hyperbolic.hyperbolic_distance(
                H.point_coordinates[i], H.point_coordinates[0])
            if ((d - matrix[i][0]) > MAX_DISTANCE_ERROR) or ((d - matrix[i][0]) < -MAX_DISTANCE_ERROR):
                print("Distance =", matrix[i][0], ", but calculated", d)
                flag += 1

    print(counter, "pair checked", flag, "pairs are wrong")
    H.print_graph("blue")


if __name__ == '__main__':
    # points()
    distance()  # FIXME ЗАМЕТИЛ ТЕНДЕНЦИЮ, ЧТО РОВНО ОДНА ПАРА ТОЧЕК СЧИТАЕТСЯ ПРАВИЛЬНО. ЗАБАВНО
    # FIXME ОЧЕНЬ РЕДКО ЧТО-ЛИБО СОВПАДАЕТ, ХОЧЕТСЯ ВЫЯСНИТЬ ПОЧЕМУ
