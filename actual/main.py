"""
главная функци с классом Hyperbolic
"""

import networkx as nx
import numpy as np
import hyperbolic
import draw
from grad_descent import GD, MSE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools


def is_on_hyperbola(point):
    print(-point[0]**2 + point[1]**2 + point[2]**2)
    return -point[0]**2 + point[1]**2 + point[2]**2 == 1.


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
        self.distances = graph
        self.__find_coordinates()  # изменяет координаты точек на гиперболоиде
        for vertex in range(graph.shape[0]):
            self.point_coordinates[vertex, 0] = np.sqrt(
                1 + sum(self.point_coordinates[vertex, 1::]**2))
        self.point_coordinates = GD(self.point_coordinates, graph, 1000)

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

    def __integral(self, p1: int, p2: int, eps=1e-6) -> np.array:
        """
        """
        distance = self.vert_dict[p1][p2]["weight"]
        v = hyperbolic.rand_vector(self.point_coordinates[p1])
        t = 0.0001
        # вершина, от которой считаем соседнюю
        domain_point = self.point_coordinates[p1]
        cur_dist = 0.

        while cur_dist <= distance:
            t *= 2
            new_ans = hyperbolic.exponential_map(
                domain_point, v, t)
            cur_dist = hyperbolic.hyperbolic_distance(
                domain_point, new_ans)

        max_t = t
        min_t = t / 2

        if abs(cur_dist - distance) < eps:
            return new_ans

        while abs(cur_dist - distance) > eps:
            t = (max_t + min_t) / 2.
            new_ans = hyperbolic.exponential_map(
                domain_point, v, t)
            cur_dist = hyperbolic.hyperbolic_distance(
                domain_point, new_ans)

            if cur_dist > distance:
                max_t = t
            elif cur_dist <= distance:
                min_t = t

        new_ans[0] = np.sqrt(1 + sum(new_ans[1:]**2))

        return new_ans

    def print_graph(self, colour='blue'):
        draw.printing(self.vert_dict, hyperbolic.projection(
            self.point_coordinates), colour)

    def draw(self, draw_eges: bool = True):
        coordinates = self.point_coordinates
        projected_coordinates = hyperbolic.projection(coordinates)

        x = projected_coordinates[:, 0]
        y = projected_coordinates[:, 1]

        # нормировка точек
        # x = x / np.linalg.norm(x)
        # y = y / np.linalg.norm(y)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.scatter(x, y)

        patch = patches.Circle((0., 0.), 1., edgecolor='black', fill=False)
        ax.add_patch(patch)

        if draw_eges:
            # отрисовка ребер графа
            for i, p1 in enumerate(zip(x, y)):
                for j, p2 in enumerate(zip(x, y)):
                    if self.distances[i, j] != 0.:
                        x_coordinates = (p1[0], p2[0])
                        y_coordinates = (p1[1], p2[1])
                        plt.plot(x_coordinates, y_coordinates,
                                 color='black')

        n = coordinates.shape[0]
        text = range(1, n + 1)
        for i, txt in enumerate(text):
            # подпись к точкам
            ax.annotate(txt, (x[i], y[i]), fontsize=12)
        plt.show()


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

# H0 = Hyperbolic(matrix, dimension)
# H0.draw()

matrix2 = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]], dtype=float)
# matrix2 = matrix2 / np.linalg.norm(matrix2)

H = Hyperbolic(graph=matrix2, dimension=2)
H.draw()
coordinates = H.point_coordinates
print("MSE: %f" % MSE(coordinates, matrix2))
# graph = nx.read_edgelist("/home/azat/Downloads/facebook/polblogs-edgelist.txt")
# graph.add_nodes_from(
#     '/home/azat/Downloads/facebook/polblogs-polblogs-nodelist.txt')
# matrix = nx.to_numpy_array(graph)
# H = Hyperbolic(matrix[:10, :10], 2)
# H.draw()
# H.draw()
