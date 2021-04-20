import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from hyperbolic import projection


def draw(coordinates: np.array, distances: np.array, draw_edges: bool = True,
         annotate: bool = False, map: dict = None):
    """
    Функция рисования проекций точек на диске Пуанкаре.

    Параметры:
    __________
    coordinates : np.array
        Координаты точек в H(n)

    distances : np.array shape = (n, n)
        Матрица смежности графа

    draw_edges : bool
        Рисовать ли ребра графа.

    annotate : bool
        Нумеровать ли точки.

    map : dict
        Для окрашивания точек.

    """
    projected_coordinates = projection(coordinates)

    x = projected_coordinates[:, 0]
    y = projected_coordinates[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    n = len(coordinates)

    half_n = int(n / 2) + 1
    if map is None:
        plt.scatter(x, y, marker='.')
    else:
        for i in range(n):
            if map[i] >= half_n:
                plt.scatter(x[i], y[i], color='blue', marker='.')
            else:
                plt.scatter(x[i], y[i], color='red', marker='.')
    if draw_edges:
        # отрисовка ребер графа
        for i, p1 in enumerate(zip(x, y)):
            for j, p2 in enumerate(zip(x, y)):
                if distances[i, j] != 0.:
                    x_coordinates = (p1[0], p2[0])
                    y_coordinates = (p1[1], p2[1])
                    plt.plot(x_coordinates, y_coordinates,
                             color='black', alpha=0.05)

    patch = patches.Circle((0, 0), radius=1.,
                           edgecolor='black', fill=False)
    ax.add_patch(patch)

    if annotate:
        n = coordinates.shape[0]
        text = range(1, n + 1)
        for i, txt in enumerate(text):
            # подпись к точкам
            ax.annotate(txt, (x[i], y[i]), fontsize=12)
    plt.savefig("image.pdf")
