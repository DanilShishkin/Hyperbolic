import main
import hyperbolic
import numpy as np


def points():
    NUM_OF_TESTS = 10
    MIN_NUM_OF_POINT = 2
    MAX_NUM_OF_POINTS = 7
    MIN_DIMENSION = 2
    MAX_DIMENSION = 5
    MAX_POINT_ERROR = 1  # FIXME СТРАННО, НО В КАКИХ-ТО СЛУЧАЯХ ЕСТЬ ОШИБКИ. НУЖНО ВЫЯСНИТЬ ПОЧЕМУ
    MIN_DISTANCE = 1
    MAX_DISTANCE = 20
    flag = 0
    counter = 0
    for test in range(NUM_OF_TESTS):
        N = np.random.randint(MIN_NUM_OF_POINT, MAX_NUM_OF_POINTS + 1, size=1, dtype=int)
        N = N[0]
        a = MIN_DISTANCE + np.random.ranf([N, N]) * MAX_DISTANCE
        matrix = np.tril(a) + np.tril(a, -1).T
        for e in range(len(matrix[0])):
            matrix[e][e] = 0
        dimension = np.random.randint(MIN_DIMENSION, MAX_DIMENSION + 1, size=1, dtype=int)
        dimension = dimension[0]
        H = main.Hyperbolic(matrix, dimension)
        for point in H.point_coordinates:
            counter += 1
            ans = sum(map(lambda x: x ** 2, point))
            ans -= 2 * point[0] ** 2
            if (ans + 1 >= MAX_POINT_ERROR) or (ans + 1 < -MAX_POINT_ERROR):
                print("Point", point, "not in H+")
                flag += 1

    print(counter, "points checked", flag, "points are wrong")


def distance():
    NUM_OF_TESTS = 10
    MIN_NUM_OF_POINT = 2
    MAX_NUM_OF_POINTS = 7
    MIN_DIMENSION = 2
    MAX_DIMENSION = 5
    MAX_DISTANCE_ERROR = 1
    MIN_DISTANCE = 1
    MAX_DISTANCE = 20
    flag = 0
    counter = 0
    for test in range(NUM_OF_TESTS):
        N = np.random.randint(MIN_NUM_OF_POINT, MAX_NUM_OF_POINTS + 1, size=1, dtype=int)
        N = N[0]
        a = MIN_DISTANCE + np.random.ranf([N, N]) * MAX_DISTANCE
        matrix = np.tril(a) + np.tril(a, -1).T
        for e in range(1, len(matrix[0])):
            for k in range(1, len(matrix[0])):
                matrix[e][k] = 0
        matrix[0][0] = 0
        dimension = np.random.randint(MIN_DIMENSION, MAX_DIMENSION + 1, size=1, dtype=int)
        dimension = dimension[0]
        H = main.Hyperbolic(matrix, dimension)
        for i in range(1, len(H.point_coordinates)):
            counter += 1
            d = hyperbolic.hyperbolic_distance(H.point_coordinates[i], H.point_coordinates[0])
            if ((d - matrix[i][0]) > MAX_DISTANCE_ERROR) or ((d - matrix[i][0]) < -MAX_DISTANCE_ERROR):
                print("Distance =", matrix[i][0], ", but calculated", d)
                flag += 1

    print(counter, "pair checked", flag, "pairs are wrong")


if __name__ == '__main__':
    points()
    # distance()  # FIXME ЗАМЕТИЛ ТЕНДЕНЦИЮ, ЧТО РОВНО ОДНА ПАРА ТОЧЕК СЧИТАЕТСЯ ПРАВИЛЬНО. ЗАБАВНО
    # FIXME ОЧЕНЬ РЕДКО ЧТО-ЛИБО СОВПАДАЕТ, ХОЧЕТСЯ ВЫЯСНИТЬ ПОЧЕМУ
