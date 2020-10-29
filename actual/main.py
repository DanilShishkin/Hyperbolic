#!/usr/bin/python3
import networkx as nx
import numpy as np
import math


class Hyperbolic:
    """
    Класс для работы с гиперболическим пространством. На данный момент не имеет публичных методов, кроме конструктора.
    """

    def __init__(self, graph, dimension):
        
        """
        Конструктор класса
        :param graph: на данный момент является матрицей смежности графа, в будущем может быть изменено для работы с
        другой библиотекой
        :param dimension: размерность гиперболического пространства, в котором мы планируем работать
        :return: По окончание создания объекта класса объект будет иметь доступ к изначально переданным данным,
        таким как размерность пространства и матрица смежности исходного графа,
        и будут вычислены координаты всех точек графа, перенесённых в гиперболическое пространство point_coordinates.
        """
        self.graph = graph
        self.dimension = dimension  # тут мы записали переданные данные
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.point_coordinates[0][dimension] = 1
        self.vert_dict = nx.from_numpy_array(graph)
        self.check = np.zeros(len(self.graph), dtype=int) 
        self.__find_coordinates()
        # тут нужно коротко или отдельным методом написать обход графа

    def __recursive(self, current):
        for child in self.vert_dict[current]:
            if not self.check[child]:
                #считаем расстояние
                print(*self.__integral(self.__rand_vector(current), current, child))
                self.check[child] = 1
                self.__recursive(child)

        # тут нужно коротко или отдельным методом написать обход графа

    def __find_coordinates(self):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        ВАЖНО, ЧТОБЫ КООРДИНАТЫ ТОЧКИ point_num БЫЛИ ВЫЧИСЛЕНЫ К ЭТОМУ МОМЕНТУ
        :return: ничего не возвращает, в ходе своей работы записывает вычисленные координаты в point_coordinates
        """
        self.check[0] = 1
        self.__recursive(0)

    def __rand_vector(self, point: int) -> list:
        """
        вычисляет рандомный вектор размерности n+1, т.к. модель пространства имеет размерность на 1 больше, чем исходное
        такой, что он принадлежит касательному подпространству в точке point
        :type point: int
        :rtype: list
        :return: возвращает вычисленный вектор
        """
        ans_0 = np.random.random_sample(self.dimension + 1)
        n = self.point_coordinates[point][self.dimension]
        tmp = 0
        for i in range(self.dimension + 1):
            tmp += ans_0[i] * (ans_0[i] - self.point_coordinates[point][i])
        ans_1 = 0.5 * n + math.sqrt(n / 4 + tmp)
        return list(ans_0 + [ans_1])

    def __projection(self, coords: list) -> list:  # скорее всего это не понадобится, думаю удалим
        """
        Кажется, что оно нам вряд ли понадобиться, но пусть пока тут лежит
        (в идеале)Вычисляет проекцию точки на n-мерное гиперболическое пространство
        (на данный момент вычисляет последнюю координату таким образом, чтобы, не изменяя остальные,
         сделать точку принадлежащей гиперболическому пространству.)
        :param coords: координаты точки, для которой необходимо вычислить проекцию
        :return: возвращает вычисленные координаты
        """
        last = (sum(((coords[:self.dimension]) ** 2)) + 1) ** 0.5
        return list([coords[i] for i in range(self.dimension)] + [last])

    def __integral(self, v: list, p1: int, p2: int) -> list:
        """
        считаем интеграл самым простым способом, просто двигаясь вдоль линии с малым шагом. Возвращает найденные
        координаты точки p2, но на самом деле момжно записать их в нужную ячейку прямо в этом методе, если понадобиться
        :param v: вектор
        :param p1: номер первой точки
        :param p2: номер второй точки
        :return: возвращает координаты второй точки
        :rtype: list
        """
        distance = self.graph[p1][p2]
        integral = 0
        t = 0
        dt = 0.000001
        ans = self.point_coordinates[p1]
        while integral < distance:
            t += dt
            new_ans = self.__current_coordinates(v, t, p1)
            integral += self.__distance(ans, new_ans)
            ans = new_ans
        return ans

    def __distance(self, p1: list, p2: list) -> float:
        """
        ищет расстояние между двумя точками в терминах нашей метрики. считаем, что расстояние малое и так делать
        действительно можно
        :param p1: координаты первой точки
        :param p2: коориданты второй точки
        :return: расстояние м ними
        """
        d = 0
        for i in range(self.dimension):
            d += p1[i] ** 2 - p2[i] ** 2
        d -= p1[self.dimension] ** 2 - p2[self.dimension] ** 2
        return math.sqrt(abs(d))

    def __current_coordinates(self, v: list, t: float, start_point: int) -> list:
        """
        эта штука занимается постоянным пересчётом координат искомой точки
        :rtype: list
        """
        n_v = sum(map(lambda i: i * i, v))
        ans1 = [math.cosh(n_v * t) * p for p in self.point_coordinates[start_point]]
        ans2 = [math.sinh(n_v * t) / n_v * vi for vi in v]
        ans = [a1 + a2 for a1, a2 in zip(ans1, ans2)]
        return ans



a = Hyperbolic(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]), 3)
print(*a.point_coordinates)
