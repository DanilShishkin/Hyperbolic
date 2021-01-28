"""
главная функци с классом Hyperbolic
"""

import networkx as nx
import numpy as np
import hyperbolic
import draw


class Hyperbolic:
    """
    Класс для работы с гиперболическим пространством.
    На данный момент не имеет публичных методов, кроме конструктора.
    """

    def __init__(self, graph: np.ndarray, dimension: int):
        """
        Конструктор класса
        Создаёт поле с координатами точек point_coordinates, которое заполнится
        в конце работы конструктора
        так же создаёт словарь связей координат для удобства работы
        """
        # print(graph)  # FIXME DEBUG
        self.dimension = dimension
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.vert_dict = nx.from_numpy_array(graph)
        self.__find_coordinates()
        print("OK")

    def __find_coordinates(self):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        """
        self.point_coordinates[0][0] = 1
        check = np.zeros(len(self.vert_dict), dtype=int)
        check[0] = 1
        self.__recursive(0, check)

    def __recursive(self, current: int, check: np.array):
        """
        обход графа в глубину с проверкой на то, что точка уже не вычислена
        в ходе работы записывает вычисленные координаты в массив координат
        """
        for child in self.vert_dict[current]:
            if not check[child]:
                self.point_coordinates[child] = self.__integral(current, child)
                check[child] = 1
                self.__recursive(child, check)

    def __integral(self, p1: int, p2: int) -> np.array:
        """
        считаем интеграл самым простым способом, просто двигаясь вдоль линии с малым шагом. Возвращает найденные
        координаты точки p2, но на самом деле момжно записать их в нужную ячейку прямо в этом методе, если понадобится
        """
        distance = self.vert_dict[p1][p2]["weight"]
        v = hyperbolic.rand_vector(self.point_coordinates[p1])
        integral = 0
        t = 0
        dt = 0.00001
        ans = self.point_coordinates[p1]
        while integral < distance:
            t += dt
            new_ans = hyperbolic.exponential_map(self.point_coordinates[p1], v, t)
            integral += hyperbolic.distance_pseudo_euclidean(ans, new_ans)
            ans = new_ans
        return ans  # если я ничего не путаю, то это первая точка, для которой расстояние больше заданного

    def print_graph(self, colour):
        draw.printing(self.vert_dict, hyperbolic.projection(self.point_coordinates), colour)


matrix = np.array([[0, 3, 2, 5, 0, 0, 0, 0, 0, 0],
                   [3, 0, 0, 0, 8, 6, 0, 0, 0, 0],
                   [2, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                   [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 6, 0, 0, 0, 0, 0, 0, 1, 2],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]])
dimension = 2
"""
H0 = Hyperbolic(matrix, dimension)
H0.print_graph('green')
H1 = Hyperbolic(matrix, dimension)
H1.print_graph('red')
H2 = Hyperbolic(matrix, dimension)
H2.print_graph('yellow')
H3 = Hyperbolic(matrix, dimension)
H3.print_graph('blue')
"""
matrix2 = np.array([[0, 5, 0, 0, 0, 0],
                    [5, 0, 5, 0, 0, 0],
                    [0, 5, 0, 5, 0, 0],
                    [0, 0, 5, 0, 5, 0],
                    [0, 0, 0, 5, 0, 5],
                    [0, 0, 0, 0, 5, 0]])
H = Hyperbolic(matrix2, dimension)
H.print_graph('red')
