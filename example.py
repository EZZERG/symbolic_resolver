from sympy import Symbol, MatrixSymbol, symbols
from block_matrix import BlockMatrix

# Create dimension symbol
n = symbols('n')
inner_dim = n - 1

# Create symbolic matrices with abstract dimension n
m1 = MatrixSymbol('A', inner_dim, inner_dim)  # (n-1)x(n-1) symbolic matrix
m2 = MatrixSymbol('b', inner_dim, 1)          # (n-1)x1 symbolic matrix
m3 = MatrixSymbol('c', 1, inner_dim)          # 1x(n-1) symbolic matrix
m4 = Symbol('d')                              # scalar symbol

# Create block matrix
M = BlockMatrix(m1, m2, m3, m4, n=n)

# Create another symbolic block matrix
p1 = MatrixSymbol('P', inner_dim, inner_dim)
p2 = MatrixSymbol('q', inner_dim, 1)
p3 = MatrixSymbol('r', 1, inner_dim)
p4 = Symbol('s')

N = BlockMatrix(p1, p2, p3, p4, n=n)

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