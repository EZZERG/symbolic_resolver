from sympy import Matrix, Symbol, MatrixSymbol, MatMul, BlockMatrix as SymBlockMatrix, UnevaluatedExpr, Mul, Add, trace, symbols
from typing import Union, Optional

def matrix_to_scalar(expr):
    """Convert a 1x1 matrix multiplication result to a scalar."""
    if isinstance(expr, MatMul):
        if expr.shape == (1, 1):
            return expr[0]  # Get the scalar value
    return expr

class BlockMatrix:
    def __init__(self, m1: Union[Matrix, MatrixSymbol], m2: Union[Matrix, MatrixSymbol], 
                 m3: Union[Matrix, MatrixSymbol], m4: Union[Symbol, int, float], n: Optional[Symbol] = None):
        """
        Initialize a symbolic block matrix of the form:
        [[m1, m2],
         [m3, m4]]
        where:
        - m1 is (n-1) x (n-1)
        - m2 is (n-1) x 1
        - m3 is 1 x (n-1)
        - m4 is a scalar
        
        Parameters:
        - n: Symbol representing the total dimension. If None, will be created as 'n'
        """
        if n is None:
            n = symbols('n')
        
        # Get dimensions for validation
        m1_shape = m1.shape
        m2_shape = m2.shape
        m3_shape = m3.shape
        
        # Store dimension
        self.n = n
        self.inner_dim = n - 1
        
        # Validate dimensions
        if m1_shape != (self.inner_dim, self.inner_dim):
            raise ValueError(f"m1 must be {self.inner_dim}x{self.inner_dim}")
        if m2_shape != (self.inner_dim, 1):
            raise ValueError(f"m2 must be {self.inner_dim}x1")
        if m3_shape != (1, self.inner_dim):
            raise ValueError(f"m3 must be 1x{self.inner_dim}")
        
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
    
    def __add__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        """Add two block matrices."""
        if not isinstance(other, BlockMatrix):
            raise TypeError("Can only add BlockMatrix with another BlockMatrix")
        if self.n != other.n:
            raise ValueError("Matrices must have the same dimensions")
        
        return BlockMatrix(
            self.m1 + other.m1,
            self.m2 + other.m2,
            self.m3 + other.m3,
            self.m4 + other.m4,
            n=self.n
        )
    
    def __mul__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        """Multiply two block matrices."""
        if not isinstance(other, BlockMatrix):
            raise TypeError("Can only multiply BlockMatrix with another BlockMatrix")
        if self.n != other.n:
            raise ValueError("Matrices must have the same dimensions")
        
        # Block matrix multiplication formula:
        # [[a1, a2], [a3, a4]] * [[b1, b2], [b3, b4]] =
        # [[a1*b1 + a2*b3, a1*b2 + a2*b4], [a3*b1 + a4*b3, a3*b2 + a4*b4]]
        
        new_m1 = self.m1 * other.m1 + self.m2 * other.m3
        new_m2 = self.m1 * other.m2 + self.m2 * other.m4
        new_m3 = self.m3 * other.m1 + self.m4 * other.m3
        new_m4 = trace(self.m3 * other.m2) + self.m4 * other.m4
        
        return BlockMatrix(new_m1, new_m2, new_m3, new_m4, n=self.n)
    
    def inverse(self) -> 'BlockMatrix':
        """
        Compute the symbolic inverse of the block matrix using the block matrix inversion formula.
        Uses the Schur complement method.
        """
        # For a block matrix [[A, B], [C, D]], where D is scalar,
        # the inverse is given by:
        # [[A^(-1) + A^(-1)B(D - CA^(-1)B)^(-1)CA^(-1),  -A^(-1)B(D - CA^(-1)B)^(-1)],
        #  [-(D - CA^(-1)B)^(-1)CA^(-1),                   (D - CA^(-1)B)^(-1)]]
        
        A_inv = self.m1**(-1)  # Symbolic inverse
        B = self.m2
        C = self.m3
        D = self.m4
        
        # Compute Schur complement: D - CA^(-1)B
        schur = D - trace(C * A_inv * B)  # C * A_inv * B is a scalar
        schur_inv = 1 / schur
        
        # Compute the blocks of the inverse
        inv_m1 = A_inv + schur_inv * A_inv * B * C * A_inv
        inv_m2 = -schur_inv * A_inv * B
        inv_m3 = -schur_inv * C * A_inv
        inv_m4 = schur_inv
        
        return BlockMatrix(inv_m1, inv_m2, inv_m3, inv_m4, n=self.n)
    
    def __str__(self) -> str:
        """String representation of the block matrix."""
        return f"BlockMatrix(\n{self.m1},\n{self.m2},\n{self.m3},\n{self.m4})"
    
    def _repr_latex_(self) -> str:
        """
        LaTeX representation for Jupyter notebook display.
        Converts the BlockMatrix to a SymPy BlockMatrix for rendering.
        """
        # Convert to SymPy BlockMatrix
        sym_matrix = SymBlockMatrix([[self.m1, self.m2], [self.m3, Matrix([self.m4])]])
        return sym_matrix._repr_latex_()
    
    def to_sympy(self) -> SymBlockMatrix:
        """Convert to a SymPy BlockMatrix."""
        return SymBlockMatrix([[self.m1, self.m2], [self.m3, Matrix([self.m4])]]) 