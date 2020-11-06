import math
import networkx as nx
import numpy as np


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
        print(graph)
        self.graph = graph
        self.dimension = dimension
        self.point_coordinates = np.zeros((len(graph), self.dimension + 1))
        self.vert_dict = nx.from_numpy_array(graph)
        self.__find_coordinates()

    def __recursive(self, current: int, check: np.array):
        """
        обход графа в глубину с проверкой на то, что точка уже не вычислена
        в ходе работы записывает вычисленные координаты в массив координат
        """
        for child in self.vert_dict[current]:
            if not check[child]:
                v = self.__rand_vector(current)
                self.point_coordinates[child] = self.__integral(v, current, child)
                check[child] = 1
                self.__recursive(child, check)

    def __find_coordinates(self):
        """
        Функция предназначена для поиска координат всех точек, смежных с переданной и не вычисленных ранее.
        """
        self.point_coordinates[0][self.dimension] = 1
        check = np.zeros(len(self.graph), dtype=int)
        check[0] = 1
        self.__recursive(0, check)

    def __rand_vector(self, point: int) -> np.ndarray:
        """
        это функция должна возвращать случайный вектор,
        который находится в касательном подпространстве в точке point_coordinates[point].
        """
        ans = 100 * np.random.uniform(-100, 100, self.dimension + 1)  # -100 и 100 границы генерируемых чисел
        xn = 0
        for i in range(self.dimension):
            xn += self.point_coordinates[point][i] * (ans[i] - self.point_coordinates[point][i])
        xn /= self.point_coordinates[point][self.dimension]
        xn += self.point_coordinates[point][self.dimension]
        ans[self.dimension] = xn
        for i in range(self.dimension + 1):
            ans[i] -= self.point_coordinates[point][i]
        return ans

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
        while integral < distance:
            t += dt
            new_ans = self.__current_coordinates(v, t, p1)
            integral += self.__distance(ans, new_ans)
            ans = new_ans
        return ans  # если я ничего не путаю, то это первая точка, для которой расстояние больше заданного

    def __distance(self, p1: np.array, p2: np.array) -> float:
        """
        ищет расстояние между двумя точками в терминах нашей метрики.
        """
        d = 0
        for i in range(self.dimension):
            d += (p2[i] - p1[i])**2
        d -= (p2[self.dimension] - p1[self.dimension])**2
        return np.sqrt(d)

    def __current_coordinates(self, v: np.array, t: float, start_point: int) -> np.array:
        """
        эта штука занимается постоянным пересчётом координат искомой точки
        """
        n_v = sum(map(lambda i: i * i, v))  # FIXME
        # мне не очень нравится, что я сначала прибавляю квадрат числа, а потом
        # дважды вычитаю его. Может знаете, как это пофиксить?
        n_v = math.sqrt(n_v - 2 * (v[self.dimension]) ** 2)
        ans1 = np.array([math.cosh(n_v * t) * p for p in self.point_coordinates[start_point]])
        ans2 = np.array([math.sinh(n_v * t) / n_v * vi for vi in v])
        return ans1 + ans2


a = Hyperbolic(np.array([[0, 10, 7], [10, 0, 0], [7, 0, 0]]), 3)
print(*a.point_coordinates)
