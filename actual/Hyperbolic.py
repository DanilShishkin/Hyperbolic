#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import numpy as np
import hyperbolic
from grad_descent import GD


class Hyperbolic:
    """
    Класс для работы с гиперболическим пространством.
    На данный момент не имеет публичных методов, кроме конструктора.
    """

    def __init__(self, graph: np.ndarray, dimension: int,
                 maxiter: int = 5000, batch: float = 1.):
        """
        Конструктор класса
        Создаёт поле с координатами точек point_coordinates, которое заполнится
        в конце работы конструктора
        так же создаёт словарь связей координат для удобства работы
        """
        self.dimension = dimension
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.vert_dict = nx.from_numpy_array(graph)
        self.distances = graph
        self.__find_coordinates()  # изменяет координаты точек на гиперболоиде
        for vertex in range(graph.shape[0]):
            self.point_coordinates[vertex, 0] = np.sqrt(
                1 + sum(self.point_coordinates[vertex, 1::]**2))
        self.point_coordinates = GD(
            self.point_coordinates, graph, maxiter, batch=batch)

    def __find_coordinates(self):
        """
        Функция предназначена для поиска координат всех точек,
        смежных с переданной и не вычисленных ранее.
        """
        self.point_coordinates[0][0] = 1
        check = np.zeros(len(self.vert_dict), dtype=int)
        check[0] = 1
        self.__recursive(0, check)

    def __recursive(self, current: int, check: np.array):
        """
        обход графа в глубину с проверкой на то, что точка еще не вычислена
        в ходе работы записывает вычисленные координаты в массив координат

        Параметры:
        ___________
        current : int
            Номер вершины, от которой считаем координаты соседней точки

        check : np.ndarray
            Массив из 0 и 1 для проверки пройдена ли вершина
        """
        for child in self.vert_dict[current]:
            if not check[child]:
                self.point_coordinates[child] = self.__integral(current, child)
                check[child] = 1
                self.__recursive(child, check)

    def __integral(self, p1: int, p2: int, eps: float = 1e-2) -> np.array:
        """
        Функция находит координаты точки на гиперболоиде от точки p1,
        на основании расстояния от точки p1 до p2.

        Параметры:
        __________
        p1 : int
            Номер вершины, от которой считаются новые координаты
        p2 : int
            Номер точки, координаты которой обновляются
        eps : float
            Точность

        Возвращает
        __________

        ans : np.ndarray
            Координаты точки p2 на гиперболоиде

        """
        distance = self.vert_dict[p1][p2]["weight"]
        v = hyperbolic.rand_vector(self.point_coordinates[p1])
        t = 0.01
        # вершина, от которой считаем соседнюю
        domain_point = self.point_coordinates[p1]
        cur_dist = 0.
        ans = hyperbolic.exponential_map(domain_point, v, t)

        # while cur_dist <= distance:
        #     t *= 2
        #     ans = hyperbolic.exponential_map(
        #         domain_point, v, t)
        #     cur_dist = hyperbolic.hyperbolic_distance(
        #         domain_point, ans)

        # max_t = t
        # min_t = t / 2

        # if abs(cur_dist - distance) < eps:
        #     return ans

        # while abs(cur_dist - distance) > eps:
        #     t = (max_t + min_t) / 2.
        #     ans = hyperbolic.exponential_map(
        #         domain_point, v, t)
        #     cur_dist = hyperbolic.hyperbolic_distance(
        #         domain_point, ans)

        #     if cur_dist > distance:
        #         max_t = t
        #     elif cur_dist <= distance:
        #         min_t = t

        ans[0] = np.sqrt(1 + sum(ans[1:]**2))
        return ans
