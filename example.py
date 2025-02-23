from sympy import MatrixSymbol, Symbol
from block_matrix import BlockMatrix

# Create symbolic matrices for n=3 (so m1 is 2x2)
m1 = MatrixSymbol('A', 2, 2)  # 2x2 symbolic matrix
m2 = MatrixSymbol('b', 2, 1)  # 2x1 symbolic matrix
m3 = MatrixSymbol('c', 1, 2)  # 1x2 symbolic matrix
m4 = Symbol('d')              # scalar symbol

# Create block matrix
M = BlockMatrix(m1, m2, m3, m4)

# Create another symbolic block matrix
p1 = MatrixSymbol('P', 2, 2)
p2 = MatrixSymbol('q', 2, 1)
p3 = MatrixSymbol('r', 1, 2)
p4 = Symbol('s')

N = BlockMatrix(p1, p2, p3, p4)

# Perform operations
sum_matrix = M + N
product_matrix = M * N
inverse_matrix = M.inverse()

print("Sum of matrices:")
print(sum_matrix)
print("\nProduct of matrices:")
print(product_matrix)
print("\nInverse of first matrix:")
print(inverse_matrix) 