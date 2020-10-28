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
        self.point_coordinates = list([0] * len(self.graph[0]))  # заполнили все координаты точек нулями временно
        self.point_coordinates[0] = list([0] * self.dimension + [1])  # нулевую точку поместили в (0,....., 0, 1)
        # тут нужно коротко или отдельным методом написать обход графа

    def __find_coordinates(self, point_num):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        ВАЖНО, ЧТОБЫ КООРДИНАТЫ ТОЧКИ point_num БЫЛИ ВЫЧИСЛЕНЫ К ЭТОМУ МОМЕНТУ
        :param point_num: номер точки, для которой будут вычислены координаты смежных
        :return: ничего не возвращает, в ходе своей работы записывает вычисленные координаты в point_coordinates
        """
        # нужно переписать метод так, чтобы он проверял, связаны ли точки. Возможно тут так же придётся записывать в
        # отдельную структуру данных точки, которые уже были обработаны
        j: int
        for j in range(0, len(self.point_coordinates)):  # цикл возможно тоже надо переписать
            # тут нужно будет организовать проверку того, являются ли точки смежными.
            if self.graph[point_num][j] != 0:
                v = self.__rand_vector(point_num)  # это я перепишу и вектор будет гарантированно принадлежать
                # касательному подпространству
                self.point_coordinates[j] = self.__integral(v, point_num, j)  # хз что тут не так, но запись
                # координат я починю

    def __rand_vector(self, point: int) -> list:
        """
        вычисляет рандомный вектор размерности n+1, т.к. модель пространства имеет размерность на 1 больше, чем исходное
        такой, что он принадлежит касательному подпространству в точке point
        :type point: int
        :rtype: list
        :return: возвращает вычисленный вектор
        """
        ans_0 = np.random.random_sample(self.dimension)
        n = self.point_coordinates[point][self.dimension + 1]
        tmp = 0
        for i in range(self.dimension):
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
        dt = 0.0001
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
        :return: расстояние между ними
        """
        d = 0
        for i in range(self.dimension):
            d += p1[i] ** 2 - p2[i] ** 2
        d -= p1[self.dimension + 1] ** 2 - p2[self.dimension + 1] ** 2
        return math.sqrt(d)

    def __current_coordinates(self, v: list, t: float, start_point: int) -> list:
        """
        эта штука занимается постоянным пересчётом координат искомой точки
        :rtype: list
        """
        n_v = sum(map(lambda i: i * i, v))
        ans = [math.cosh(n_v * t) * p for p in self.point_coordinates[start_point]]
        ans += [math.sinh(n_v * t) / n_v * vi for vi in v]
        print(*ans)
        return ans
