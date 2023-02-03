import numpy as np

m1 = (
    (1, -1),
    (1, 2)
)
matrix = np.array(
    m1
)
print(matrix)

output = np.array(
    (0, 8)
)

solve = np.linalg.solve(matrix, output)  # 线性方程组求解 .
print(solve)

X = np.arange(-5, 5, 0.25)
a1 = np.arange(0, 10, 0.5)
print(a1)
