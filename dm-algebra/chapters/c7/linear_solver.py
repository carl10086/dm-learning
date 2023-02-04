import numpy as np

from chapters.internal.tools.vectors import *


def standard_form(v1, v2):
    """
    1个函数表达的形式是 ax + by = c .
    我们通过代数的方式计算出来 .
    而这个 算出来的值 刚好能够被 numpy 帮助我们计算 .
    """
    x1, y1 = v1
    x2, y2 = v2
    a = y2 - y1
    b = x1 - x2
    c = x1 * y2 - y1 * x2
    return a, b, c


def intersection(u1, u2, v1, v2):
    """
    u1, u2 决定了1条直线
    v1 ,v2 决定了另外1条
    通过矩阵运算
    """
    a1, b1, c1 = standard_form(u1, u2)
    a2, b2, c2 = standard_form(v1, v2)
    m = np.array(((a1, b1), (a2, b2)))
    c = np.array((c1, c2))
    return np.linalg.solve(m, c)


## Will fail if lines are parallel!
# def do_segments_intersect(s1,s2):
#     u1,u2 = s1
#     v1,v2 = s2
#     l1, l2 = distance(*s1), distance(*s2)
#     x,y = intersection(u1,u2,v1,v2)
#     return (distance(u1, (x,y)) <= l1 and
#             distance(u2, (x,y)) <= l1 and
#             distance(v1, (x,y)) <= l2 and
#             distance(v2, (x,y)) <= l2)

def segment_checks(s1, s2):
    u1, u2 = s1
    v1, v2 = s2
    l1, l2 = distance(*s1), distance(*s2)
    x, y = intersection(u1, u2, v1, v2)
    return [
        distance(u1, (x, y)) <= l1,
        distance(u2, (x, y)) <= l1,
        distance(v1, (x, y)) <= l2,
        distance(v2, (x, y)) <= l2
    ]


def do_segments_intersect(s1, s2):
    u1, u2 = s1
    v1, v2 = s2
    d1, d2 = distance(*s1), distance(*s2)
    try:
        x, y = intersection(u1, u2, v1, v2)
        return (distance(u1, (x, y)) <= d1 and
                distance(u2, (x, y)) <= d1 and
                distance(v1, (x, y)) <= d2 and
                distance(v2, (x, y)) <= d2)
    except np.linalg.linalg.LinAlgError:
        return False

# print(do_segments_intersect(((0,2),(1,-1)),((0,0),(4,0))))

# a = np.array(((1,0), (0,1)))
# b = np.array((9,0))
# x = np.linalg.solve(a, b)
# print(x)
