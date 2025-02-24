from sympy import Symbol, MatrixSymbol, symbols
from block_matrix import BlockMatrix, BlockVector

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

# Create block vector components
v_fix = MatrixSymbol('v', inner_dim, 1)      # (n-1)x1 vector
x = Symbol('x')                              # scalar for last element

# Create block vector
v = BlockVector(v_fix, x, n=n)

print("Original Block Matrix M:")
print(M)
print("\nOriginal Block Vector v:")
print(v)

# Test scalar operations
alpha = Symbol('alpha')  # Symbolic scalar for multiplication
beta = Symbol('beta')   # Symbolic scalar for addition

# Scalar operations with vector
print("\nScalar operations with vector:")
print("\n1. Scalar multiplication (alpha * v):")
print(alpha * v)
print("\n2. Scalar multiplication (v * alpha):")
print(v * alpha)
print("\n3. Scalar addition (beta + v):")
print(beta + v)
print("\n4. Scalar addition (v + beta):")
print(v + beta)

# Scalar operations with matrix
print("\nScalar operations with matrix:")
print("\n1. Scalar multiplication (alpha * M):")
print(alpha * M)
print("\n2. Scalar multiplication (M * alpha):")
print(M * alpha)

# Test combined operations
print("\nCombined operations:")
print("\n1. Matrix-vector product with scaled vector (M * (alpha * v)):")
print(M * (alpha * v))
print("\n2. Scaled matrix-vector product ((alpha * M) * v):")
print((alpha * M) * v)

# Create second vector for testing
w_fix = MatrixSymbol('w', inner_dim, 1)
y = Symbol('y')
w = BlockVector(w_fix, y, n=n)

print("\nVector operations with scalars:")
print("\n1. Inner product of scaled vectors:")
print("Inner product of (alpha * v) and w:")
print((alpha * v).inner_product(w))

# Test matrix operations with scalars
print("\nMatrix operations with scalars:")
N = BlockMatrix(m1, m2, m3, m4, n=n)  # Create a copy of M
print("\n1. Product of scaled matrices:")
print("(alpha * M) * (beta * N):")
print((alpha * M) * (beta * N))

# Let's also test vector addition and inner product
v2_fix = MatrixSymbol('w', inner_dim, 1)
y = Symbol('y')
v2 = BlockVector(v2_fix, y, n=n)

# Vector addition
sum_vec = v + v2

# Inner product
inner_prod = v.inner_product(v2)

print("\nSecond Block Vector v2:")
print(v2)
print("\nVector sum v + v2:")
print(sum_vec)
print("\nInner product <v, v2>:")
print(inner_prod)

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