import math
import networkx as nx
import numpy as np


class Hyperbolic:
    """
    Класс для работы с гиперболическим пространством. На данный момент не имеет публичных методов, кроме конструктора.
    """

    def __init__(self, graph: np.ndarray, dimension: int):
        """
        Конструктор класса
        Создаёт поле с координатами точек point_coordinates, которое заполнится
        в конце работы конструктора
        так же создаёт словарь связей координат для удобства работы
        """
        self.graph = graph
        self.dimension = dimension
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.vert_dict = nx.from_numpy_array(graph)
        print(graph)
        self.__find_coordinates()

    def __recursive(self, current: int, check: np.array):
        """
        обход графа в глубину с проверкой на то, что точка уже не вычислена
        в ходе работы записывает вычисленные координаты в массив координат
        """
        for child in self.vert_dict[current]:
            if not check[child]:
                self.point_coordinates[child] = self.__integral(self.__rand_vector(current), current, child)
                check[child] = 1
                self.__recursive(child, check)

    def __find_coordinates(self):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        ВАЖНО, ЧТОБЫ КООРДИНАТЫ ТОЧКИ point_num БЫЛИ ВЫЧИСЛЕНЫ К ЭТОМУ МОМЕНТУ
        """
        self.point_coordinates[0][self.dimension] = 1
        check = np.zeros(len(self.graph), dtype=int)
        check[0] = 1
        self.__recursive(0, check)

    def __rand_vector(self, point: int) -> np.ndarray:
        """
        это функция должна возвращать случайный вектор,
        который находится в касательном подпространстве в точке point_coordinates[point]
        пока я заменил это всё заглушкой. Она будет работать, но только с теми входными данными,
        где все точки связаны только с нулевой.
        """
        return np.array([1 + point, 1, 0])

    def __integral(self, v: np.array, p1: int, p2: int) -> np.array:
        """
        считаем интеграл самым простым способом, просто двигаясь вдоль линии с малым шагом. Возвращает найденные
        координаты точки p2, но на самом деле момжно записать их в нужную ячейку прямо в этом методе, если понадобится
        """
        distance = self.graph[p1][p2]
        integral = 0
        t = 0
        dt = 0.0001
        ans = self.point_coordinates[p1]
        while integral < distance ** 2:  # пока что мы считаем, что расстояние может быть комплексным
            # и сравниваем сумму квадратов элементарных расстояний с квадратом заданной дистанции.
            # Есть подозрение, что это неправда и необходимо как-то пофиксить данную часть
            t += dt
            new_ans = self.__current_coordinates(v, t, p1)
            integral += self.__distance(ans, new_ans)
            ans = new_ans
        return ans  # если я ничего не путаю, то это первая точка, для которой расстояние больше заданного

    def __distance(self, p1: np.array, p2: np.array) -> float:
        """
        ищет расстояние между двумя точками в терминах нашей метрики.
        считаем, что расстояние мало и так делать действительно можно.
        """
        d = 0
        for i in range(self.dimension):
            d += p1[i] ** 2 - p2[i] ** 2
        d -= p1[self.dimension] ** 2 - p2[self.dimension] ** 2
        return d  # по-хорошему расстоянием является корень этого значения,
        # но он иногда отрицательный по неизвестным причинам

    def __current_coordinates(self, v: np.array, t: float, start_point: int) -> np.array:
        """
        эта штука занимается постоянным пересчётом координат искомой точки
        """
        n_v = sum(map(lambda i: i * i, v))  # FIXME
        # мне не очень нравится, что я сначала прибавляю квадрат числа, а потом
        # дважды вычитаю его. Может знаете, как это пофиксить?
        n_v = math.sqrt(math.fabs(n_v - 2 * (v[self.dimension]) ** 2))
        ans1 = np.array([math.cosh(n_v * t) * p for p in self.point_coordinates[start_point]])
        ans2 = np.array([math.sinh(n_v * t) / n_v * vi for vi in v])
        return ans1 + ans2

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


a = Hyperbolic(np.array([[0, 10, 1], [10, 0, 0], [1, 0, 0]]), 2)
print(*a.point_coordinates)
