from tkinter import Tk, Canvas, Frame, BOTH


class Example(Frame):

    def __init__(self, vert_dict, point_coordinates, colour):
        super().__init__()
        self.initUI(vert_dict, point_coordinates, colour)

    def initUI(self, vert_dict, coord, colour):
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        canvas.create_oval(0, 800, 800, 0)
        for i in range(len(coord)):
            canvas.create_oval(
                400 + coord[i][0] - 5, 400 + coord[i][1] +
                5, 400 + coord[i][0] + 5, 400 + coord[i][1] - 5,
                outline=colour, fill=colour, width=1
            )
        # for current in vert_dict:
        #     for child in vert_dict[current]:
        #         canvas.create_line(400 + coord[current][0], 400 + coord[current][1],
        #                            400 + coord[child][0], 400 + coord[child][1])
        canvas.pack(fill=BOTH, expand=1)


def printing(vert_dict, point_coordinates, colour):
    root = Tk()
    ex = Example(vert_dict, point_coordinates, colour)
    root.geometry("400x100+300+300")
    root.mainloop()
