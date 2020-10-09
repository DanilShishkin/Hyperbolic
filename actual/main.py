import numpy as np


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
        self.dimension = dimension
        self.point_coordinates = list([0] * len(self.graph[0]))
        self.point_coordinates[0] = tuple([0] * self.dimension + [1])
        for i in range(len(self.point_coordinates)):
            self.__find_coordinates(i)

    def __find_coordinates(self, point_num):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        Поэтому в конструкторе передаются все точки в порядке возрастания,
        а вычисляются точки с номером не менее переданного.
        :param point_num: номер точки, для которой будут вычислены координаты смежных
        :return: ничего не возвращает, в ходе своей работы записывает вычисленные координаты в point_coordinates
        """
        for j in range(point_num + 1, len(self.point_coordinates)):
            # тут нужно будет организовать проверку того, являются ли точки смежными.
            v = self.__rand_vector()
            magic = v  # тут когда-нибудь произойдёт магия с вычислением координат точки
            # на данный момент вместо нужных точек вычисляется точка, заданная проекциями рандомных радиус-векторов
            self.point_coordinates[j] = self.__projection(magic)

    def __rand_vector(self):
        """
        вычисляет рандомный вектор размерности n+1, т.к. модель пространства имеет размерность на 1 больше, чем исходное
        :return: возвращает вычисленный вектор
        """
        return np.random.random_sample(self.dimension + 1)

    def __projection(self, coords):
        """
        (в идеале)Вычисляет проекцию точки на n-мерное гиперболическое пространство
        (на данный момент вычисляет последнюю координату таким образом, чтобы, не изменяя остальные,
         сделать точку принадлежащей гиперболическому пространству.)
        :param coords: координаты точки, для которой необходимо вычислить проекцию
        :return: возвращает вычисленные координаты
        """
        last = ((sum(coords[:self.dimension])) + 1)**0.5
        return tuple([coords[i] for i in range(self.dimension)] + [last])


# G = ((0, 1, 0), (1, 0, 1), (0, 1, 0), (1, 1, 1))  # Тестовый случай, просто две связанные между собой точки
# a = Hyperbolic(G, 3)  # Вызываем конструктор класса
