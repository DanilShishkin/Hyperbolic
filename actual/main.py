#import numpy  # думаю нам это понадобится


class Hyperbolic:   # основной класс, с ним мы будем работать. На вход он принимает уже созданный граф, скорее всего он
                    # будет выглядеть просто как матрица смежности. Так же при инициализации

    def __init__(self, graph):
        self.graph = graph
        self.point_coordinates = list()  # координаты точек, в конце инициализации будут записаны в этот список
        self.point_coordinates[0]  # тут мы присвоим нашей первой точке координаты 0,....,0,1
        for i in range (self.point_coordinates.len()):
            # для каждой точки из нашего графа
            self.find_coordinates(i) # ищем координаты всех смежных с ней точек, кроме тех, что уже найдены

    def _find_coordinates_(self, point_num):
        for j in range (point_num + 1, self.point_coordinates.len()):
            # все ненайденные точки
            V = self.rand_vector()  # рандомное направление
            magic = 0  # тут происходит магия с вычислением координат точки
            self.point_coordinates[j] = self.projection(magic)

    def _rand_vector_(self):  # эта функция должна вернуть рандомный вектор
        pass

    def _projection_(self, coords):  # возвращает проекцию точки на гиперболическое пространство
        pass


