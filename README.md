# Hyperbolic
Чтобы использовать модуль **Hyperbolic** необходимо подключить его (*import Hyperbolic*). Далее для использования необходимо создать класс Hyperbolic, передать в конструктор 
* матрицу смежности взвешенного графа (матрица np.array размера (n, n), n - количество вершин).
* Передать размерность гиперболического пространства, в которое вкладывается граф (dimension). 
* Параметр maxiter - количество итераций градиентного спуска, 
* bathc - доля точек, от которых считается градиент.

Для работы модуля необходимо, чтобы расстояния между вершинами графа были не больше 20. Можно использовать нормировку матрицы смежности.

Далее для построения на диске Пуанкаре точек можно подключить модуль **draw**, вызвать функцию draw - в нее передаются 
* координаты точек - (матрица размера (n, dimension+1)). 
* distance - матрица смежности, которая передается в конструктор класса Hyperbolic. 
* draw_edges - рисовать ли ребра графа. 
* annotate - подписывать ли номера точек. 
* map - словарь : {номер точки при пересавлении ее номера -> номер в изначальном графе}.

Так же можно написать самостоятельно отрисовку графа. Для этого необходимо проецировать точки из гиперболического пространства на диск Пуанкаре. Для этого в модуле **hyperbolic** есть функция *projection*, в нее передаются координаты точек на гиперболоиде (метод *point_coordinates* класса Hyperbolic возвращает эти координаты).
*hyperbolic.projection* возвращает *np.array* размера (n, dimension).
# Пример использования
```python
from Hyperbolic import Hyperbolic
import draw
distance = np.array(size=(n, n)) # матрица смежности 
# Вложение графа distance в гиперболическое пространство размерности 2
# С 1000 итерациями градиентного спуска, градиент считается от 10% точек
H = Hyperbolic(graph=distance, dimension=2, maxiter=1000, batch=0.1) 

# Рисование точек графа на диске Пуанкаре
# Без построения ребер графа
draw(coordinates = H.point_coordinates, distances=distance, draw_edges=False)
```
