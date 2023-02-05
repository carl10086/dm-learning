from math import sqrt, ceil, floor

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlim, ylim

blue = 'C0'
black = 'k'
red = 'C3'
green = 'C2'
purple = 'C4'
orange = 'C2'
gray = 'gray'

dino_vectors = [
    (6, 4),
    (3, 1),
    (1, 2),
    (-1, 5),
    (-2, 5),
    (-3, 4),
    (-4, 4),
    (-5, 3),
    (-5, 2),
    (-2, 2),
    (-5, 1),
    (-4, 0),
    (-2, 1),
    (-1, 0),
    (0, -3),
    (-1, -4),
    (1, -4),
    (2, -3),
    (1, -2),
    (3, -1),
    (5, 1),
]


class Polygon2D:

    def __init__(self, *vertices, color=blue, fill=None, alpha=0.4):
        self.vertices = vertices
        self.color = color
        self.fill = fill
        self.alpha = alpha


class Points2D():
    def __init__(self, *vectors, color=black):
        self.vectors = list(vectors)
        self.color = color


class Arrow2D:
    def __init__(self, tip, tail=(0, 0), color=red):
        self.tip = tip
        self.tail = tail
        self.color = color


class Segment2D:
    def __init__(self, start_point, end_point, color=blue):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color


# helper function to extract all the vectors from a list of objects
def extract_vectors(objects):
    for item in objects:
        if type(item) == Polygon2D:
            for v in item.vertices:
                yield v
        elif type(item) == Points2D:
            for v in item.vectors:
                yield v
        elif type(item) == Arrow2D:
            yield item.tip
            yield item.tail
        elif type(item) == Segment2D:
            yield item.start_point
            yield item.end_point
        else:
            raise TypeError("Unrecognized object: {}".format(item))


def draw2d(*objects, origin=True, axes=True, grid=(1, 1), nice_aspect_ratio=True, width=6, save_as=None):
    all_vectors = list(extract_vectors(objects))
    xs, ys = zip(*all_vectors)

    max_x, max_y, min_x, min_y = max(0, *xs), max(0, *ys), min(0, *xs), min(0, *ys)

    # sizing
    if grid:
        x_padding = max(ceil(0.05 * (max_x - min_x)), grid[0])
        y_padding = max(ceil(0.05 * (max_y - min_y)), grid[1])

        def round_up_to_multiple(val, size):
            return floor((val + size) / size) * size

        def round_down_to_multiple(val, size):
            return -floor((-val - size) / size) * size

        plt.xlim(floor((min_x - x_padding) / grid[0]) * grid[0],
                 ceil((max_x + x_padding) / grid[0]) * grid[0])
        plt.ylim(floor((min_y - y_padding) / grid[1]) * grid[1],
                 ceil((max_y + y_padding) / grid[1]) * grid[1])

    if origin:
        plt.scatter([0], [0], color='k', marker='x')

    if grid:
        plt.gca().set_xticks(np.arange(plt.xlim()[0], plt.xlim()[1], grid[0]))
        plt.gca().set_yticks(np.arange(plt.ylim()[0], plt.ylim()[1], grid[1]))
        plt.grid(True)
        plt.gca().set_axisbelow(True)

    if axes:
        plt.gca().axhline(linewidth=2, color='k')
        plt.gca().axvline(linewidth=2, color='k')

    for item in objects:
        if type(item) == Polygon2D:
            for i in range(0, len(item.vertices)):
                x1, y1 = item.vertices[i]
                x2, y2 = item.vertices[(i + 1) % len(item.vertices)]
                plt.plot([x1, x2], [y1, y2], color=item.color)
            if item.fill:
                xs = [v[0] for v in item.vertices]
                ys = [v[1] for v in item.vertices]
                plt.gca().fill(xs, ys, item.fill, alpha=item.alpha)
        elif type(item) == Points2D:
            xs = [v[0] for v in item.vectors]
            ys = [v[1] for v in item.vectors]
            plt.scatter(xs, ys, color=item.color)
        elif type(item) == Arrow2D:
            tip, tail = item.tip, item.tail
            tip_length = (xlim()[1] - xlim()[0]) / 20.
            length = sqrt((tip[1] - tail[1]) ** 2 + (tip[0] - tail[0]) ** 2)
            new_length = length - tip_length
            new_y = (tip[1] - tail[1]) * (new_length / length)
            new_x = (tip[0] - tail[0]) * (new_length / length)
            plt.gca().arrow(tail[0], tail[1], new_x, new_y,
                            head_width=tip_length / 1.5, head_length=tip_length,
                            fc=item.color, ec=item.color)
        elif type(item) == Segment2D:
            x1, y1 = item.start_point
            x2, y2 = item.end_point
            plt.plot([x1, x2], [y1, y2], color=item.color)
        else:
            raise TypeError("Unrecognized object: {}".format(item))

    fig = matplotlib.pyplot.gcf()

    if nice_aspect_ratio:
        coords_height = (ylim()[1] - ylim()[0])
        coords_width = (xlim()[1] - xlim()[0])
        fig.set_size_inches(width, width * coords_height / coords_width)

    if save_as:
        plt.savefig(save_as)

    plt.show()
