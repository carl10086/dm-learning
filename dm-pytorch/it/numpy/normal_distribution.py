import numpy as np
import matplotlib.pyplot as plt

A = np.array(
    [
        [56.0, 0.0, 4.4, 68.0],
        [1.2, 104.0, 52.0, 8.0],
        [1.8, 135.0, 99.0, 0.9],
    ]
)

print(A.shape)

print(A.sum(axis=0))
print(A.sum(axis=1))

cal = A.sum(axis=0)

percentage = A / cal

print(percentage)
