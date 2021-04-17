import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from hyperbolic import projection


def draw(coordinates: np.array, draw_eges: bool = True,
         annotate: bool = False, map: dict = None):
    """
    Функция рисования проекций точек на диске Пуанкаре.

    Параметры:
    __________
    coordinates : np.array
        Координаты точек в H(n)

    draw_edges : bool
        Рисовать ли ребра графа.

    annotate : bool
        Нумеровать ли точки.

    map : dict
        Для окрашивания точек.
    """
    projected_coordinates = projection(coordinates)

    x = projected_coordinates[:, 0] * 100.
    y = projected_coordinates[:, 1] * 100.

    fig, ax = plt.subplots()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    n = len(coordinates)

    half_n = int(n / 2) + 1
    for i in range(n):
        if map[i] >= half_n:
            plt.scatter(x[i], y[i], color='blue')
        else:
            plt.scatter(x[i], y[i], color='red')

    patch = patches.Circle((0, 0), radius=1.,
                           edgecolor='black', fill=False)
    ax.add_patch(patch)

    if annotate:
        n = coordinates.shape[0]
        text = range(1, n + 1)
        for i, txt in enumerate(text):
            # подпись к точкам
            ax.annotate(txt, (x[i], y[i]), fontsize=12)
    plt.show()
