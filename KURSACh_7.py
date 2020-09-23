from copy import deepcopy
from decimal import Decimal


# import add_func
# import networkx as nx


class Hyperbolic(object):
    def __init__(self, x0, y0, length):
        self.x0 = x0  # объявление переменных
        self.y0 = y0
        self.length = length

    @property
    def z0(self):  # вычисление z, чтобы она точно лежала на поверхности
        return ((self.x0 ** 2) + (self.y0 ** 2) + 1) ** (1 / 2)

    def x1_y (self, A, B, C , y):
        return((-A * B * y + C * (A**2 - C**2 + y**2 *(A**2 + B**2 - C**2))**(0.5))/(A**2 - C**2))

    def x2_y (self, A, B, C , y):
        return((-A * B * y - C * (A**2 - C**2 + y**2 *(A**2 + B**2 - C**2))**(0.5))/(A**2 - C**2))

    def y1_x (self, A, B, C , x):
        return((-A * B * x + C * (B**2 - C**2 + x**2 *(A**2 + B**2 - C**2))**(0.5))/(B**2 - C**2))

    def y2_x(self, A, B, C, x):
        return ((-A * B * x - C * (B**2 - C**2 + x**2 * (A**2 + B**2 - C**2))**(0.5))/(B**2 - C**2))


    def z_demox(self, A, B, C, x):
        return ((x**2 + self.x1_y(A,B,C,x)**2 + 1)**(0.5))



    """""
    def x_t(self, x_s, A, B, C, t):  # вычисляем х от t и у от
        if x_s >= 0:
            ansx = (-A * C * t + (-A ** 2 * B ** 2 + A ** 2 * B ** 2 * t ** 2 + B ** 4 * t ** 2 - B ** 2 * C ** 2 * t ** 2) ** (0.5)) / (A ** 2 + B ** 2)
        elif x_s < 0:
            ansx = (-A * C * t - (-A ** 2 * B ** 2 + A ** 2 * B ** 2 * t ** 2 + B ** 4 * t ** 2 - B ** 2 * C ** 2 * t ** 2) ** (0.5)) / (A ** 2 + B ** 2)
        return (ansx)

    def y_t(self, y_s, A, B, C, t):
        if y_s >= 0:
            ansy = (-B * C * t + A*(-A ** 2 + B ** 2 + A ** 2 * t ** 2 + B ** 2 * t ** 2 - C ** 2 * t ** 2) ** (0.5)) / (A ** 2 + B ** 2)
        elif y_s < 0:
            ansy = (-B * C * t - A*(-A ** 2 + B ** 2 + A ** 2 * t ** 2 + B ** 2 * t ** 2 - C ** 2 * t ** 2) ** (0.5)) / (A ** 2 + B ** 2)
        return (ansy)
"""
    @property #что это?
    def hyperbolic_point(self):

        print('Input vector')
        x_s = float(
            input())  # эти переменные мы запоминаем для того, чтобы знать, в каком направлении мы двигаемся
        y_s = float(input())

        x1 = x_s + self.x0  # вычисляем смещение от нашей точки по заданному вектору
        y1 = y_s + self.y0
        z1 = float(input()) + self.z0
        A = -self.z0 * y1 + self.y0 * z1  # Коэффициенты нормали нашей плоскости
        B = self.z0 * x1 - self.x0 * z1
        C = self.x0 * y1 - x1 * self.y0

        integral = 0
        t = self.x0
        step = 0.000001

        while self.length - integral > 0:
            t = t + step
            integral += (step ** 2 + (self.y1_x(A, B, C, t) - self.y1_x(A, B, C, (t-step))) ** 2 - (self.z_demox(A, B, C, t) - self.z_demox(A,B,C, t-step)) ** 2) ** (0.5)

        x_ans = t
        print("x = ", t)
        y_ans = self.y1_x(A, B, C, t)
        print("y = ", y_ans)
        z_ans = self.z_demox(A, B, C, t)
        print("z = ", z_ans)
        return (t)


if __name__ == '__main__':
    print('Input a point (x, y) and length')
    xi = float(input())
    yi = float(input())
    lengthi = float(input())

    point = Hyperbolic(xi, yi, lengthi)
    ans = point.hyperbolic_point()

"""a= 0.1
    b = 0.2
    x = 0.001
    c = 0.0000001
    first = Decimal(((b ** 2) * ((a ** 2) * (x ** 2) + (b ** 2) * ((x ** 2) + 1) - (c ** 2) * ((x ** 2) + 1))) ** 0.5)
    print(first)"""
# print ((0.1 ** 2) * (0.1 ** 2))
