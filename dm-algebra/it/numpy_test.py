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


