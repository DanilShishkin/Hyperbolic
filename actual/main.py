"""
главная функци с классом Hyperbolic
"""

import networkx as nx
import numpy as np
import hyperbolic
import draw
from grad_descent import GD


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
        self.dimension = dimension
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.vert_dict = nx.from_numpy_array(graph)
        self.__find_coordinates()  # изменяет координаты точек на гиперболоиде
        print(graph)
        print(self.point_coordinates)
        for vertex in range(graph.shape[0]):
            self.point_coordinates[vertex, 0] = np.sqrt(
                1 + sum(self.point_coordinates[vertex, 1::]**2))
        self.point_coordinates = GD(self.point_coordinates, graph, 100)

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
        обход графа в глубину с проверкой на то, что точка уже не вычислена
        в ходе работы записывает вычисленные координаты в массив координат
        """
        for child in self.vert_dict[current]:
            if not check[child]:
                self.point_coordinates[child] = self.__integral(current, child)
                check[child] = 1
                self.__recursive(child, check)

    def __integral(self, p1: int, p2: int, eps=1e-1) -> np.array:
        """
        """
        distance = self.vert_dict[p1][p2]["weight"]
        v = hyperbolic.rand_vector(self.point_coordinates[p1])
        integral = 0.0  # расстояние на гиперболоиде
        t = 0.001
        ans = self.point_coordinates[p1]

        while abs(distance - integral) > eps:
            t *= 2
            new_ans = hyperbolic.exponential_map(
                self.point_coordinates[p1], v, t)
            cur_dist = hyperbolic.hyperbolic_distance(ans, new_ans)

            if integral + cur_dist > distance:
                t = t / 2 + 0.0001
                new_ans = hyperbolic.exponential_map(
                    self.point_coordinates[p1], v, t)
                cur_dist = hyperbolic.hyperbolic_distance(ans, new_ans)
                integral += cur_dist
                ans = new_ans
                continue
            else:
                integral += cur_dist
                ans = new_ans
        return ans

    def print_graph(self, colour):
        draw.printing(self.vert_dict, hyperbolic.projection(
            self.point_coordinates), colour)

# matrix = np.array([[0, 3, 2, 5, 0, 0, 0, 0, 0, 0],
#                    [3, 0, 0, 0, 8, 6, 0, 0, 0, 0],
#                    [2, 0, 0, 0, 0, 0, 1, 2, 0, 0],
#                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 6, 0, 0, 0, 0, 0, 0, 1, 2],
#                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]])
# dimension = 2

# H0 = Hyperbolic(matrix, dimension)
# H0.print_graph('green')
# H1 = Hyperbolic(matrix, dimension)
# H1.print_graph('red')
# H2 = Hyperbolic(matrix, dimension)
# H2.print_graph('yellow')
# H3 = Hyperbolic(matrix, dimension)
# H3.print_graph('blue')


# matrix2 = np.array([[0, 1, 1],
#                     [1, 0, 1],
#                     [1, 1, 0]], dtype=float)

# H = Hyperbolic(graph=matrix2, dimension=2)
# H.print_graph('blue')
