import sympy as sp
from sympy import Matrix, BlockMatrix, MatrixSymbol, Symbol, ZeroMatrix

def assert_block_matrix_shape(M, name="matrix"):
    """Assert that a block matrix has the expected shape with blocks:
    [[m1 (n-1,n-1), m2 (n-1,1)],
     [m3 (1,n-1),   m4 (1,1)]]
    """
    if not isinstance(M, BlockMatrix):
        raise TypeError(f"{name} must be a BlockMatrix")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]
    
    # Check that m1 is (n-1, n-1)
    if m1.shape[0] != m1.shape[1]:
        raise ValueError(f"{name} block m1 must be square")
    
    n = m1.shape[0] + 1  # Total size
    
    # Check remaining shapes
    if m2.shape != (n-1, 1):
        raise ValueError(f"{name} block m2 must be shape ({n-1}, 1)")
    if m3.shape != (1, n-1):
        raise ValueError(f"{name} block m3 must be shape (1, {n-1})")
    if m4.shape != (1, 1):
        raise ValueError(f"{name} block m4 must be shape (1, 1)")

def assert_block_vector_shape(V, name="vector"):
    """Assert that a block vector has the expected shape:
    [[v_fix (n-1,1)],
     [last_element (1,1)]]
    """
    if not isinstance(V, BlockMatrix):
        raise TypeError(f"{name} must be a BlockMatrix")
    
    if len(V.blocks) != 2 or V.blocks[0, 0].shape[1] != 1 or V.blocks[1, 0].shape != (1, 1):
        raise ValueError(f"{name} must be a column vector with shape (n, 1)")
    
    v_fix = V.blocks[0, 0]
    last_element = V.blocks[1, 0]
    
    if last_element.shape != (1, 1):
        raise ValueError(f"{name} last element must be scalar (1,1)")

def create_block_vector(last_element_symbol, V_fix_symbol='V_{n-1}'):
    """Create a block vector with symbolic blocks
    
    Args:
        last_element_symbol: Symbol for the last element
        V_fix_symbol: Symbol for the V_fix block (default: 'V_{n-1}')
    """
    # Create a symbolic size variable
    n = Symbol('n')
    # Create V_fix as a matrix symbol with symbolic size
    V_fix = MatrixSymbol(V_fix_symbol, n, 1)
    # Create the last element
    last_element = MatrixSymbol(last_element_symbol, 1, 1)
    # Combine into a block vector
    V = BlockMatrix([[V_fix], [last_element]])
    assert_block_vector_shape(V, "result")
    return V

def create_block_matrix():
    """Create a block matrix with symbolic blocks"""
    # Create a symbolic size variable
    n = Symbol('n')
    # Create blocks with symbolic dimensions
    k1 = MatrixSymbol('k1', n, n)
    k2 = MatrixSymbol('k2', n, 1)
    k3 = MatrixSymbol('k3', 1, n)
    k4 = MatrixSymbol('k4', 1, 1)
    
    # Combine into block matrix
    K = BlockMatrix([[k1, k2], [k3, k4]])
    assert_block_matrix_shape(K, "result")
    return K

def create_block_diagonal_matrix(fix_block_symbol, last_block_symbol):
    """Create a block diagonal matrix with symbolic blocks"""
    n = Symbol('n')
    fix_block = MatrixSymbol(fix_block_symbol, n, n)
    last_block = MatrixSymbol(last_block_symbol, 1, 1)
    
    # Use ZeroMatrix instead of MatrixSymbol('0', n, 1)
    result = BlockMatrix([[fix_block, ZeroMatrix(n, 1)], 
                         [ZeroMatrix(1, n), last_block]])
    assert_block_matrix_shape(result, "result")
    return result

def block_matrix_vector_product(M, V):
    """Compute the block matrix-vector product explicitly"""
    assert_block_matrix_shape(M, "M")
    assert_block_vector_shape(V, "V")
    
    k1 = M.blocks[0, 0]
    k2 = M.blocks[0, 1]
    k3 = M.blocks[1, 0]
    k4 = M.blocks[1, 1]
    
    V_fix = V.blocks[0, 0]
    x = V.blocks[1, 0]
    
    upper_block = k1 * V_fix + k2 * x
    lower_block = k3 * V_fix + k4 * x
    
    return BlockMatrix([[upper_block], [lower_block]])

def block_matrix_matrix_addition(M, N):
    """Compute the block matrix-matrix addition explicitly"""
    assert_block_matrix_shape(M, "M")
    assert_block_matrix_shape(N, "N")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]

    n1 = N.blocks[0, 0]
    n2 = N.blocks[0, 1]
    n3 = N.blocks[1, 0]
    n4 = N.blocks[1, 1]

    # Simplify each block sum
    return BlockMatrix([[sp.simplify(m1 + n1), sp.simplify(m2 + n2)], 
                       [sp.simplify(m3 + n3), sp.simplify(m4 + n4)]])

def block_vector_vector_subtraction(M, N):
    """Compute the block vector-vector subtraction explicitly"""
    assert_block_vector_shape(M, "M")
    assert_block_vector_shape(N, "N")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[1, 0]
    
    n1 = N.blocks[0, 0]
    n2 = N.blocks[1, 0]

    result = BlockMatrix([[sp.simplify(m1 - n1)], 
                         [sp.simplify(m2 - n2)]])
    assert_block_vector_shape(result, "result")
    return result

def scalar_block_matrix_product(scalar, M):
    """Compute the scalar-block matrix product explicitly"""
    assert_block_matrix_shape(M, "M")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]

    result = BlockMatrix([[sp.simplify(scalar * m1), sp.simplify(scalar * m2)], 
                         [sp.simplify(scalar * m3), sp.simplify(scalar * m4)]])
    assert_block_matrix_shape(result, "result")
    return result

def block_matrix_matrix_product(M, N):
    """Compute the block matrix-matrix product explicitly"""
    assert_block_matrix_shape(M, "M")
    assert_block_matrix_shape(N, "N")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]

    n1 = N.blocks[0, 0]
    n2 = N.blocks[0, 1]
    n3 = N.blocks[1, 0]
    n4 = N.blocks[1, 1]

    first_block = sp.simplify(m1 * n1 + m2 * n3)
    second_block = sp.simplify(m1 * n2 + m2 * n4)
    third_block = sp.simplify(m3 * n1 + m4 * n3)
    fourth_block = sp.simplify(m3 * n2 + m4 * n4)

    result = BlockMatrix([[first_block, second_block], 
                         [third_block, fourth_block]])
    assert_block_matrix_shape(result, "result")
    return result

def inverse_block_matrix(M):
    """Compute the inverse of a block matrix
    M = [[m1, m2], [m3, m4]]
    We assume m1 is invertible and m4 is scalar
    
    For diagonal matrices where m2=m3=0, we compute inverse directly
    """
    assert_block_matrix_shape(M, "M")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]
    
    # Extract m4 as scalar symbol directly (since we know it's 1x1)
    if isinstance(m4, MatrixSymbol):
        m4_scalar = sp.Symbol(m4.name)  # Create a scalar symbol with the same name
    else:
        m4_scalar = m4[0, 0]
    
    # Check if matrix is diagonal (m2 and m3 are zero matrices)
    if isinstance(m2, ZeroMatrix) and isinstance(m3, ZeroMatrix):
        result = BlockMatrix([[m1.I, ZeroMatrix(m1.shape[0], m4.shape[1])],
                          [ZeroMatrix(m4.shape[0], m1.shape[1]), sp.simplify(m4_scalar**(-1)) * Matrix([[1]])]])
    else:
        m1_inv = m1.I
        # Compute Schur complement directly
        schur_scalar = m4_scalar - sp.Symbol('k3k1invk2', commutative=True)
        s_inv_scalar = sp.simplify(schur_scalar**(-1))
        
        # Compute blocks directly without substitution
        block_11 = m1_inv + s_inv_scalar * (m1_inv * m2 * m3 * m1_inv)
        block_12 = -s_inv_scalar * (m1_inv * m2)
        block_21 = -s_inv_scalar * (m3 * m1_inv)
        block_22 = s_inv_scalar * Matrix([[1]])
        
        result = BlockMatrix([[block_11, block_12], [block_21, block_22]])
    
    assert_block_matrix_shape(result, "result")
    return result

def assert_block_vector_transposed_shape(V, name="vector"):
    """Assert that a transposed block vector has the expected shape:
    [v_fix.T (1,n-1), last_element.T (1,1)]
    """
    if not isinstance(V, BlockMatrix):
        raise TypeError(f"{name} must be a BlockMatrix")
    
    if len(V.blocks) != 2:
        raise ValueError(f"{name} must have exactly 2 blocks")
    
    v_fix_t = V.blocks[0]
    last_element_t = V.blocks[1]
    
    if v_fix_t.shape[0] != 1 or last_element_t.shape != (1, 1):
        raise ValueError(f"{name} must be a row vector with shape (1, n)")

def transposed_vector_matrix_product(V_T, M_T):
    """Compute V^T * M^T = (M*V)^T where V is a column vector"""
    if not isinstance(V_T, BlockMatrix) or not isinstance(M_T, BlockMatrix):
        raise TypeError("Inputs must be BlockMatrix")
    
    assert_block_vector_transposed_shape(V_T, "V_T")
    assert_block_matrix_shape(M_T, "M_T")
    
    v1_t = V_T.blocks[0]
    v2_t = V_T.blocks[1]
    
    m1_t = M_T.blocks[0, 0]
    m2_t = M_T.blocks[0, 1]
    m3_t = M_T.blocks[1, 0]
    m4_t = M_T.blocks[1, 1]
    
    upper = v1_t * m1_t + v2_t * m3_t
    lower = v1_t * m2_t + v2_t * m4_t
    
    result = BlockMatrix([[upper, lower]])
    assert_block_vector_transposed_shape(result, "result")
    return result

def transpose_matrix(M):
    """Compute the transpose of a matrix"""
    assert_block_matrix_shape(M, "M")
    
    m1 = M.blocks[0, 0]
    m2 = M.blocks[0, 1]
    m3 = M.blocks[1, 0]
    m4 = M.blocks[1, 1]
    
    result = BlockMatrix([[m1.T, m3.T], [m2.T, m4]])
    assert_block_matrix_shape(result, "result")
    return result

def transpose_vector(V):
    """Compute the transpose of a vector"""
    assert_block_vector_shape(V, "V")
    
    v1 = V.blocks[0, 0]
    v2 = V.blocks[1, 0]
    
    result = BlockMatrix([v1.T, v2.T])
    assert_block_vector_transposed_shape(result, "result")
    return result

def main():
    # Create two vectors V1 and V2 with different last elements
    V1 = create_block_vector('x1')
    V2 = create_block_vector('x2')
    print("Vector V1:")
    print(V1)
    print("\nVector V2:")
    print(V2)
    
    # Create matrix M
    M = create_block_matrix()
    print("\nMatrix M:")
    print(M)
    
    # Calculate the products M·V1 and M·V2
    result1 = block_matrix_vector_product(M, V1)
    result2 = block_matrix_vector_product(M, V2)
    print("\nProduct M·V1:")
    print(result1)
    print("\nProduct M·V2:")
    print(result2)
    
    # Create and invert a block matrix
    K = create_block_matrix()
    K_inv = inverse_block_matrix(K)
    print("\nOriginal matrix K:")
    print(K)
    print("\nInverse of K:")
    print(K_inv)
    
    # Verify that K * K^(-1) = I
    product = block_matrix_matrix_product(K, K_inv)
    print("\nVerification K * K^(-1):")
    print(product)

if __name__ == "__main__":
    main()
