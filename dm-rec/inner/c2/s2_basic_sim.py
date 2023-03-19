import numpy as np


def cn(s1, s2):
    return len(s1 & s2)


def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)


def cos4set(s1: set, s2: set):
    return len(s1 & s2) / (len(s1) * len(s2)) ** 0.5


def cos4vec(v1: np.ndarray, v2: np.ndarray):
    return v1.dot(v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
    )


def pearson(v1: np.ndarray, v2: np.ndarray):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)

    # return (np.dot(v1 - v1_mean, v2 - v2_mean)) / \
    #     (np.linalg.norm(v1 - v1_mean) *
    #      np.linalg.norm(v2 - v2_mean))
    return cos4vec(v1 - v1_mean, v2 - v2_mean)


if __name__ == '__main__':
    sa = {1, 2, 3}
    sb = {2, 3, 4}

    a = [1, 3, 2]
    b = [2, 3, 4]

    va = np.array(a)
    vb = np.array(b)

    print(
        f"""
        cn: {cn(sa, sb)}
        
        Jarcard: {jaccard(sa, sb)}
        
        cos4vec: {cos4vec(va, vb)}
        
        cos4set: {cos4set(sa, sb)}
        
        pearson: {pearson(va, vb)}
        """
    )