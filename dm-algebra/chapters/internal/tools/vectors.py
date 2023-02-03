from math import *


def to_cartesian(polar_vector):
    """
    从极坐标 到 vector
    :param polar_vector: 极坐标, 长度 + 角度(和x轴)
    :return:
    """
    _length, angle = polar_vector[0], polar_vector[1]
    return _length * cos(angle), _length * sin(angle)


def length(vector):
    """
    一个 矢量的长度
    :param vector:
    :return:
    """
    # return sqrt(vector[0] ** 2 + vector[1] ** 2)
    return sqrt(sum([v ** 2 for v in vector]))


def to_polar(vector):
    """
    从向量到极坐标
    :param vector:
    :return:
    """
    x, y = vector[0], vector[1]
    angle = atan2(y, x)
    return length(vector), angle


def add(*vectors):
    """
    add for vectors
    :param vectors:
    :return:
    """
    by_coordinate = zip(*vectors)
    coordinates = [sum(coords) for coords in by_coordinate]
    # return sum([v[0] for v in vectors]), sum([v[1] for v in vectors])
    return tuple(coordinates)


def subtract(v1, v2):
    """
    2个向量的减法
    """
    return tuple([z[0] - z[1] for z in zip(v1, v2)])


def distance(v1, v2):
    """
    2个向量的距离
    """
    return length(subtract(v1, v2))


def scale(scalar, v):
    """
    向量点积
    :param scalar:  标量
    :param v:  向量集合
    :return:
    """
    return tuple([item * scalar for item in v])


def translate(translation, vectors):
    """
    对每个 vectors 的每个 vector都做 + translation 法
    :param translation:    :param vectors:    :return:
    """
    return [add(translation, v) for v in vectors]


def rotate2d(angle, vector):
    """
    矢量坐标按照 角度 angle 进行渲染
    :param angle:  要旋转的角度
    :param vectors: 笛卡尔积坐标系下的 向量
    :return:
    """
    l, a = to_polar(vector)
    return to_cartesian((l, a + angle))


def dot(u, v):
    """
    计算2个向量的点积
    :param u: 笛卡尔坐标系下的 vector
    :param v: 笛卡尔坐标系下的 vector
    :return:
    """
    return sum([item[0] * item[1] for item in zip(u, v)])


def angle_between(v1, v2):
    """
    计算2个 向量的夹角 < 180度的 那个 .
    :param v1: 笛卡尔坐标系下的 vector
    :param v2: 笛卡尔坐标系下的 vector
    :return:
    """
    # return acos((length(v1) * length(v2)) / dot(v1, v2))
    return acos(dot(v1, v2) / (length(v1) * length(v2)))


def cross(u, v):
    """
    仅仅适合 三维向量
    :param u: 笛卡尔坐标系下的 vector
    :param v: 笛卡尔坐标系下的 vector
    :return:
    """
    ux, uy, uz = u
    vx, vy, vz = v

    return uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
