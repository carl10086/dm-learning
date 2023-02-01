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


if __name__ == '__main__':
    v = ["1", "2", "3"]
    print(",".join(v))
