#import "../book.typ": book-page

#show: book-page.with(title: "Linear Algebra")

= Matrix Computation

In this chapter, we explore fundamental operations in linear algebra, including matrix multiplication, linear systems, matrix decompositions, and eigenvalue problems. We'll provide practical examples and applications for each concept. For a comprehensive reference on matrix computations, see @Golub2016.

== Matrix Multiplication
Matrix multiplication is a core operation in linear algebra. For matrices $A in CC^(m times n)$ and $B in CC^(n times p)$, their product $C = A B$ is defined as:
$ C_(i j) = sum_(k=1)^n A_(i k) B_(k j) $

For square matrices where $n = p = m$, the standard algorithm has $O(n^3)$ time complexity. While advanced techniques like the Strassen algorithm and laser methods @Alman2021 can achieve $O(n^(2.81))$ or better, the standard algorithm remains most practical for typical matrix sizes. Modern implementations leverage hardware optimizations like tiling and parallelization to approach peak performance.

== Linear Systems and LU Decomposition
For an invertible matrix $A in CC^(n times n)$ and vector $b in CC^n$, solving a linear system means finding $x in CC^n$ such that:
$ A x = b $

#block(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Example:* Consider the system:
    $ cases(
      2x_1 + 3x_2 - 2x_3 &= 1,
      3x_1 + 2x_2 + 3x_3 &= 2,
      4x_1 - 3x_2 + 2x_3 &= 3
    ) $

    In matrix form:
    $ mat(
      2, 3, -2;
      3, 2, 3;
      4, -3, 2
    ) mat(x_1; x_2; x_3) = mat(1; 2; 3) $

    In Julia, we can solve this using the backslash operator:
    ```julia
    A = [2 3 -2; 3 2 3; 4 -3 2]
    b = [1, 2, 3]
    x = A \ b
    A * x
    ```

    The backslash method uses LU decomposition internally:
    ```julia
    using LinearAlgebra
    lures = lu(A)  # pivot rows by default
    lures.L * lures.U ≈ lures.P * A

    UpperTriangular(lures.U) \ (LowerTriangular(lures.L) \ (lures.P * b))
    ```
  ]
)

The LU decomposition of a matrix $A in CC^(n times n)$ is a factorization:
$ P A = L U $
where $P$ is a permutation matrix for pivoting rows, $L$ is lower triangular, and $U$ is upper triangular. Row pivoting ensures numerical stability by avoiding division by zero. In Julia, matrices marked as `UpperTriangular` or `LowerTriangular` are solved efficiently using forward and backward substitution.

To solve a linear system using LU decomposition:

1. Decompose $P A in CC^(n times n)$ into $L in CC^(n times n)$ and $U in CC^(n times n)$ using Gaussian elimination or Crout's method.

2. Rewrite $A x = b$ as $L U x = P b$.

3. Forward-substitution: Solve $L y = P b$ by working top-down, substituting known values.

4. Back-substitution: Solve $U x = y$ by working bottom-up, substituting known values.

== Least Squares Problem and QR Decomposition

The least squares problem seeks to minimize the residual:
$ norm(A x - b)_2 $
where $A in CC^(m times n)$ and $b in CC^m$. This is solved using QR decomposition:
$ A = Q R $
where $Q in CC^(m times m)$ is orthogonal and $R in CC^(m times n)$ is upper triangular.

#block(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Example: Data Fitting*

    Consider fitting a quadratic function $y = c_0 + c_1 t + c_2 t^2$ to data points:

    #table(
      columns: 11,
      align: center,
      [$t_i$], [0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5],
      [$y_i$], [2.9], [2.7], [4.8], [5.3], [7.1], [7.6], [7.7], [7.6], [9.4], [9.0]
    )

    #figure(
      image("images/fitting-data.png", width: 60%),
      caption: "Data points and fitted curve"
    )

    The least squares problem minimizes:
    $ sum_(i=1)^n (y_i - (c_0 + c_1 t_i + c_2 t_i^2))^2 $

    In matrix form:
    $ min_x norm(A x - b)_2 $
    where:
    $ A = mat(
      1, t_1, t_1^2;
      1, t_2, t_2^2;
      dots.v, dots.v, dots.v;
      1, t_n, t_n^2
    ),
    x = mat(c_0; c_1; c_2),
    b = mat(y_1; y_2; dots.v; y_n) $

    While we can solve this using the normal equations:
    $ x = (A^dagger A)^(-1) A^dagger b $
    this approach is numerically unstable for large matrices. Instead, use QR decomposition:

    ```julia
    Q, R = qr(A)
    x = R \ (Matrix(Q)' * y)
    ```

    Or the pseudoinverse (using SVD internally):
    ```julia
    x = LinearAlgebra.pinv(A) * y
    ```
  ]
)

== Eigenvalues and Eigenvectors
The eigenvalues and eigenvectors of a matrix $A in CC^(n times n)$ are the solutions to the equation:
$ A x = lambda x $
where $lambda$ is a scalar and $x$ is a non-zero vector. The eigenvalues of a matrix can be found by solving the characteristic equation:
$ det(A - lambda I) = 0 $
where $I$ is the identity matrix.

In Julia, we can find the eigenvalues and eigenvectors using the `eigen` function:

```julia
A = [1 2; 3 4]
eigen(A)
```

#block(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Example: Eigenmodes of a Vibrating String*
    
    Consider a one-dimensional vibrating string or atomic chain:

    #figure(
      image("images/spring.png"),
      caption: [Vibrating string model. #link("https://lampz.tugraz.at/~hadley/ss1/phonons/1d/1dphonons.php")[Image source]]
    )
    
    The dynamics follow Newton's second law:
    $ M dot.double(u) = C(u_(i+1) - u_i) - C(u_i - u_(i-1)) $
    where $M$ is mass, $C$ is stiffness, and $u_i$ is the displacement of atom $i$. With fixed ends ($u_0 = u_(n+1) = 0$) and assuming harmonic motion:
    $ u_i(t) = A_i cos(omega t + phi_i) $

    This transforms into an eigenvalue problem:
    $ mat(
      -C, C, 0, dots.h, 0;
      C, -2C, C, dots.h, 0;
      0, C, -2C, dots.h, 0;
      dots.v, dots.v, dots.v, dots.down, dots.v;
      0, 0, 0, dots.h, -C
    ) mat(
      A_1;
      A_2;
      A_3;
      dots.v;
      A_n
    ) = -omega^2 M mat(
      A_1;
      A_2;
      A_3;
      dots.v;
      A_n
    ) $

    Let's solve for a 5-atom string with $M = C = 1.0$:

    ```julia
    M = C = 1.0
    C_matrix = [-C C 0 0 0; C -2C C 0 0; 0 C -2C C 0; 0 0 C -2C C; 0 0 0 C -C]
    evals, evecs = LinearAlgebra.eigen(C_matrix)
    second_omega = sqrt(-evals[2]/M)
    second_mode = evecs[:, 2]
    u(t) = second_mode .* cos.(-second_omega .* t) # (ϕi=0)
    u(1.0)  # atom locations at t=1.0
    ```

    #figure(
      image("images/springs-demo.gif", width: 60%),
      caption: "Visualization of eigenmodes"
    )

    Any initial condition can be expressed as a linear combination of these eigenmodes. For implementation details, see the #link("https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/PhysicsSimulation")[source code].
  ]
)

== Matrix functions

For an analytic function $f$ defined by a power series and matrix $A in CC^(n times n)$:
$ f(A) = sum_(i=0)^infinity a_i A^i $

To compute a matrix function (e.g., $f(A) = e^A$):

1. Diagonalize $A = P D P^(-1)$, where $D$ is diagonal and $P$ contains eigenvectors
2. Compute $f(A) = P f(D) P^(-1)$
3. Apply $f$ to diagonal elements of $D$
4. Multiply matrices: $f(A) = P f(D) P^(-1)$

#box(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Example: Matrix Exponential*
    
    Consider the matrix:
    $ A = mat(1, 2; 3, 4) $

    Using Julia's `exp` function:
    ```julia
    A = [1 2; 3 4]
    exp(A)
    ```

    Verify using eigendecomposition:
    ```julia
    D, P = LinearAlgebra.eigen(A)
    P * LinearAlgebra.Diagonal(exp.(D)) * inv(P)
    ```
  ]
)

== Singular Value Decomposition
The singular value decomposition (SVD) of a matrix $A in CC^(m times n)$ is a factorization:
$ A = U Sigma V^dagger $
where:
- $U in CC^(m times m)$ and $V in CC^(n times n)$ are unitary matrices
- $Sigma in CC^(m times n)$ is a diagonal matrix with non-negative real numbers

The SVD generalizes eigenvalue decomposition to non-square matrices and provides a powerful tool for matrix analysis, dimensionality reduction, and solving least squares problems.

In Julia, we compute SVD using the `svd` function:

```julia
A = [1 2; 3 4; 5 6]
F = svd(A)
F.U * Diagonal(F.S) * F.Vt ≈ A  # verify decomposition
```

#box(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Applications of SVD*
    
    1. *Principal Component Analysis (PCA)*: Use left singular vectors (columns of $U$) as principal components
    2. *Image Compression*: Keep only the largest singular values and corresponding vectors
    3. *Pseudoinverse*: Compute $A^+ = V Sigma^+ U^dagger$ for least squares problems
    4. *Matrix Rank*: Count non-zero singular values
    5. *Matrix Condition Number*: Ratio of largest to smallest singular value
  ]
)

== Cholesky Decomposition
For a symmetric positive-definite matrix $A in CC^(n times n)$, the Cholesky decomposition provides a factorization:
$ A = L L^dagger $
where $L in CC^(n times n)$ is lower triangular. This decomposition is:
- Unique when $A$ is positive definite
- More efficient than LU decomposition for symmetric matrices
- Numerically stable without pivoting

In Julia:
```julia
A = [2 1; 1 3]  # symmetric positive-definite
C = cholesky(A)
C.L * C.L' ≈ A  # verify decomposition
```

#box(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Common Applications*

    1. *Linear Systems*: Solve $A x = b$ using forward and backward substitution
    2. *Monte Carlo Simulation*: Generate correlated random variables
    3. *Optimization*: Test for positive definiteness and compute search directions
    4. *Kalman Filtering*: Update covariance matrices efficiently
  ]
)

#bibliography("refs.bib")