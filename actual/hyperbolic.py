"""
Функции на H(n)
"""
import numpy as np


def scalar_product(point_1: np.array, point_2: np.array) -> float:
    """
    Скалярное произведение в гиперболическом пространстве
    -x_0 * y_0 + sum_{i=1}^{n} x_i * y_i

    Параметры:
    __________

    point_1 : np.array
        Первый вектор

    point_2 : np.array
        Второй вектор

    Возвращает:
    ___________

    scalar_product : float
        Скалярное произведение векторов point_1 и point_2
    """
    ans = point_1*point_2
    ans[0] = -ans[0]
    scalar_product = ans.sum()
    return scalar_product


def norm(v: np.array) -> float:
    """
    Ищет норму вектора в R(n+1)

    Параметры:
    __________

    v : np.array
        Вектор, у которого считается норма

    Возвращает:
    ___________

    Норму вектора v в R(n+1)
    """
    return np.sqrt(scalar_product(v, v))


def hyperbolic_distance(p1: np.array, p2: np.array) -> float:
    """
    Считает расстояние между двумя точками в H(n)

    Параметры:
    ___________
    p1 : np.array
        Первая точка
    p2 : np.array
        Вторая точка, такой же размерности как p1

    Возвращает:
    ____________
    Расстояние в H(n) по формуле arccosh(-(x, y)_L),
    где (x, y)_L = -x_0 * y_0 + sum_{i=1}^{n} x_i * y_i
    """
    inner = scalar_product(p1, p2)
    return np.arccosh(-inner)


def exponential_map(start_point: np.array, v: np.array, t: float) -> np.array:
    """
    Вычисляет координаты точки от точки start_point в направлении 
    касательного вектора v на расстоянии t.

    Параметры:
    __________

    start_point : np.array
        Точка на H(n), от которой считаем точку
    v : np.array 
        Вектор в касательном пространстве к start_point
    t : float
        Расстояние, на котором необходимо построить точку. (Длина вектора v)

    Возвращает:
    ____________

    Координаты точки на гиперболоиде.

    """
    nv = norm(v)
    return np.cosh(t) * start_point + np.sinh(t) * v / nv


def rand_vector(point: np.array) -> np.array:
    """
    Строит случайный вектор в касательном пространстве 
    к точке гиперболоида point.

    Параметры:
    __________

    point : np.array
        Координаты точки на гиперболоиде, 
        к которой необходимо найти касательный вектор

    Возвращает:
    ___________

    tang_vector : np.array
        Вектор, касательный к гиперболоиду в точке point
    """

    dimension = len(point)
    tang_vector = np.zeros(dimension)
    tang_vector[1:] = np.random.uniform(-1., 1., size=dimension - 1)
    tang_vector[0] = sum(point[1:] * tang_vector[1:]) / point[0]
    return tang_vector


def projection(coordinates: np.array):
    """
    Проекция точек на диск Пуанкаре.
    p(x0, x1, ..., xn) = (x1, x2, ..., xn) / (x0 + 1)

    Параметры:
    ___________

    coordinates : np.array, 2-dimensional
        Координаты точек на гиперболоиде

    Возвращает:
    ____________
    np.array
        Двумерный массив координат, 
        где координат на 1 меньше, чем было в coordinates.
    """

    return np.array([coordinates[i, 1:] / (coordinates[i, 0] + 1)
                     for i in range(coordinates.shape[0])])
