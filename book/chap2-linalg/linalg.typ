#import "../book.typ": book-page
#import "@preview/cetz:0.2.2": *

#show: book-page.with(title: "Linear Algebra")
#let exampleblock(it) = block(fill: rgb("#ffffff"), inset: 1em, radius: 4pt, stroke: black, it)
#set math.equation(numbering: "(1)")

= Matrix Computation

In this chapter, we explore fundamental operations in linear algebra, including matrix multiplication, linear systems, matrix decompositions, and eigenvalue problems. We'll provide practical examples and applications for each concept. For a comprehensive reference on matrix computations, see @Golub2016.

== Notations

- Real Scalar: $x in RR$
- Real Matrix: $A in RR^(m times n)$
- Complex Scalar: $z in CC$
- Transpose: $A^T$
- Complex Conjugate: $A^*$
- Hermitian conjugate: $A^dagger ("or" A^H) = (A^T)^*$
- Unitary matrix: $U^dagger U = U U^dagger = I$

== Matrix Multiplication
Matrix multiplication is a core operation in linear algebra. For matrices $A in CC^(m times n)$ and $B in CC^(n times p)$, their product $C = A B$ is defined as:
$ C_(i j) = sum_(k=1)^n A_(i k) B_(k j) $

For square matrices where $n = p = m$, the standard algorithm has $O(n^3)$ time complexity. While advanced techniques like the Strassen algorithm and laser methods @Alman2021 can achieve $O(n^(2.81))$ or better, the standard algorithm remains most practical for typical matrix sizes. Modern implementations leverage hardware optimizations like tiling and parallelization to approach peak performance.

=== Tutorial point: The Big $O$ notation and flop count

The big $O$ notation characterizes the *scaling* of the computational time with respect to the size of the input. The complexity of naive matrix multiplication is $O(n^3)$, which means the number of operations scales as $n^3$ when the size of the matrix $n$ increases.
```julia
for i in 1:n   # first loop
  for j in 1:n  # second loop
    for k in 1:n  # third loop
      A[i, j] = max(A[i, j], A[i, k] + A[k, j])  # number of operations: n^3
    end
  end
end
```
On any machine, whenever we double the size of the matrix, the number of operations will be $2^3 = 8$ times. So the big $O$ notation is a property of the algorithm, not the machine. To more accurately describe the performance of an algorithm, we can use the *flop* (floating-point operation) count. For example, the flop count of matrix multiplication is $2 n^3$, as we have $n^2$ multiplications and $n^2$ additions. The flop count considers the basic operations, including addition, subtraction, multiplication, and division.

== Linear Systems and LU Decomposition
Let $A in RR^(n times n)$ be an invertible square matrix and $b in RR^n$ be a vector. Solving a linear equation means finding a vector $x in RR^n$ such that
$
A x = b
$

#exampleblock([
    *Example:* Let us consider the following system of linear equations
    $ cases(
      2x_1 + 3x_2 - 2x_3 &= 1,
      3x_1 + 2x_2 + 3x_3 &= 2,
      4x_1 - 3x_2 + 2x_3 &= 3
    ) $

    The matrix form of the system is
    $
    A x = b\
    A = mat(2, 3, -2; 3, 2, 3; 4, -3, 2), quad
    x = vec(x_1, x_2, x_3), quad
    b = vec(1, 2, 3)
    $



    In Julia, we can solve this using the "`\`" operator:
    ```julia
    A = [2 3 -2; 3 2 3; 4 -3 2]
    b = [1, 2, 3]
    x = A \ b
    A * x
    ```
  ]
)


In the above example, when applied on a square matrix, the "`\`" operator uses LU decomposition internally:
```julia
using LinearAlgebra
lures = lu(A)  # pivot rows by default
lures.L * lures.U ≈ lures.P * A

UpperTriangular(lures.U) \ (LowerTriangular(lures.L) \ (lures.P * b))
```

The LU decomposition of a matrix $A in CC^(n times n)$ is a factorization:
$ P A = L U $
where $P$ is a permutation matrix for pivoting rows, $L$ is lower triangular, and $U$ is upper triangular. Row pivoting ensures numerical stability by avoiding division by zero. In Julia, matrices marked as `UpperTriangular` or `LowerTriangular` are solved efficiently using forward and backward substitution.

To solve a linear system using LU decomposition:

1. Decompose $P A in CC^(n times n)$ into $L in CC^(n times n)$ and $U in CC^(n times n)$ using Gaussian elimination or Crout's method.

2. Rewrite $A x = b$ as $L U x = P b$.

3. Forward-substitution: Solve $L y = P b$ by working top-down, substituting known values.

4. Back-substitution: Solve $U x = y$ by working bottom-up, substituting known values.

== Least Squares Problem and QR Decomposition

The least squares problem is to find a vector $x in RR^n$ that minimizes the residual
$
min_x norm(A x - b)_2
$ <eq:lsq-problem>
where $A in RR^(m times n)$ and $b in RR^m$.

A linear equation is just a special case of the least squares problem, where the *residual* is zero.
The least squares problem "makes sense" only when $A$ is *over-determined* (meaning having too many equations such that not all can be satisfied), i.e. $m > n$.

Converting the least squares problem to a normal equation is a straightforward approach to solve it.
We first square and expand the residual in @eq:lsq-problem:
$
(A x - b)^T (A x - b) = x^T A^T A x - 2 x^T A^T b + b^T b.
$
The minimum is attained when the gradient of the quadratic function is zero, i.e.
$
nabla_(x) (x^T A^T A x - 2 x^T A^T b + b^T b) = 2 A^T A x - 2 A^T b = 0\
arrow.double.r x = (A^T A)^(-1) A^T b,
$
which is the normal equation.

== Sensitivity of the normal equation
== Floating-point numbers and relative errors

Layout for 64-bit floating point numbers (IEEE 754 standard)

#figure(canvas({
  import draw: *
  let (dx, dy) = (1.0, 1.0)
  let s(it) = text(16pt)[#it]
  content((-1, 0), s[$x:$])
  rect((-dx/2, -dy/2), (dx/2, dy/2), name: "sign")
  rect((dx/2, -dy/2), (3.5*dx, dy/2), name: "exponent")
  rect((3.5*dx, -dy/2), (6*dx, dy/2), name: "mantissa")
  content("sign", s[$plus.minus$])
  content("exponent", s[$a_1 a_2 dots a_11$])
  content("mantissa", s[$b_1 b_2 dots b_52$])
  content((rel: (0, -1), to: "sign"), s[sign])
  content((rel: (0, -1), to: "exponent"), s[exponent])
  content((rel: (0, -1), to: "mantissa"), s[mantissa])
}),
)
It represents the number $x = plus.minus 1.b_1 b_2 dots b_52 2^(a_1 a_2 dots a_11 - 1023)$.

== Floating point errors

$"fl"(x) = x (1 + delta), quad |delta| <= u$, where $u$ is the unit roundoff defined by
$
u = "nextfloat"(1.0) - 1.0
$


#box(text(16pt)[```julia
julia> eps(Float64)  # 2.22e-16
julia> eps(1e-50)    # absolute precision: 1.187e-66
julia> eps(1e50)     # absolute precision: 2.077e34
julia> typemax(Float64)  # Inf
julia> prevfloat(Inf)    # 1.798e308
```])

- Q: why is the relative precision $u$ remains approximately the same for different numbers?

== Stability of floating-point arithmetic
Task: compute $p - sqrt(p^2 + q)$ for $p = 12345678$ and $q = 1$.

- Method 1:
#box(text(16pt)[```julia
p - sqrt(p^2 + q)       # -4.0978193283081055e-8
```])
- Method 2:
#box(text(16pt)[```julia
-q/(p + sqrt(p^2 + q))  # -4.0500003321000205e-8
```])

Q: which one is more accurate? Hint: imagine we perform "$-$" operation on two very close large numbers.

== Condition number

- _Condition number_ of a matrix $A$ is defined as $kappa(A) = ||A|| ||A^(-1)|| >=1$. If the condition number is close to 1, the matrix is _well-conditioned_, otherwise it is _ill-conditioned_.

- _Remark_: there are two popular norms for matrices: the Frobenius norm and the $p$-norms. Here, we use the $p=2$ norm for simplicity.
  - Frobenius norm: $||A||_F = sqrt(sum_(i j) |a_(i j)|^2)$
  - $p$-norm: $||A||_p = max_(x != 0) (||A x||_p)/ (||x||_p)$, for $p = 2$, it is the spectral norm $||A||_2 = sigma_1(A)$, the largest _singular value_ of $A$.

== Meaning of the condition number

- _Remark_: meaning of the condition number: if we solve the linear system $A x = b$ with a small perturbation $b + delta b$, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

$
  A(x + delta x) = b + delta b\
  arrow.double.r (||delta x||) / (||x||) = (||A^(-1) delta b||) / (||A^(-1) b||) <= (lambda_1(A^(-1))) / (lambda_n (A^(-1))) (||delta b||) / (||b||)
$
where $lambda_1(A^(-1))$ and $lambda_n (A^(-1))$ ($= lambda_1(A)^(-1)$) are the largest and smallest _singular values_ of $A^(-1)$, respectively.

Hence, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

== Singular values decomposition
The singular values decomposition (SVD) of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = U S V^dagger
$
where $U in bb(C)^(m times m)$ and $V in bb(C)^(n times n)$ are unitary matrices (i.e. $U^dagger U = I$ and $V^dagger V = I$), and $S = "diag"(lambda_1, lambda_2, dots, lambda_n)$ is a diagonal matrix with *non-negative* real numbers on the diagonal.

- _Remark_: the SVD is a generalization of the eigendecomposition of a matrix. The diagonal elements of $S$ are the singular values arranged in descending order.
- _Remark_: For real matrices, $U$ and $V$ are orthogonal matrices (i.e. $U^T U = I$ and $V^T V = I$).

== SVD and condition number

Consider $A = U S V^dagger$,
$
  (||A x||_2) / (||x||_2) = (||S V^dagger x||_2) / (||x||_2) = (||S y||_2) / (||y||_2) <= lambda_1,
$
where $y = V^dagger x$. We used the fact that $||U x||_2 = ||x||_2$ for any unitary matrix $U$.

The effect of "squaring" a matrix:
$
  A^dagger A = V S^dagger U^dagger U S V^dagger = V S^2 V^dagger.
$
The singular values of $A^dagger A$ are the squared singular values of $A$.

Hence, $kappa(A^dagger A) = kappa(A)^2$.

== Example: condition number of the normal equation

Problem: The condition number of $A^dagger A$ is the square of the *condition number* of $A$, which can be very large.

#box(text(16pt)[```julia
julia> cond(A)
34.899220365288556

julia> cond(A' * A)
1217.9555821049864
```])

Revisit the normal equation:
$
x = (A^T A)^(-1) A^T b
$
We effectively solve the linear system: $(A^T A) x = A^T b$, which is unstable.



== QR Decomposition
This is solved using QR decomposition:
$ A = Q R $
where $Q in CC^(m times m)$ is orthogonal and $R in CC^(m times n)$ is upper triangular.

#exampleblock([
    *Example: Data Fitting*

Consider we have a set of data points $(t_i, y_i)$ for $i = 1, 2, dots, n$.
The objective of the data fitting problem is to find a *smooth* curve that fits the data the *best*.
#figure(table(columns: 11,
[$t_i$], [0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5],
[$y_i$], [2.9], [2.7], [4.8], [5.3], [7.1], [7.6], [7.7], [7.6], [9.4], [9.0],
))

#figure(canvas(length:0.9cm, {
  import plot
  import draw: *
  let t = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
  let y = (2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4, 9.0)
  plot.plot(size: (10,6),
    x-tick-step: 1,
    y-tick-step: 2,
    x-label: [$t$],
    y-label: [$y$],
    y-max: 10,
    y-min: 0,
    x-max: 5,
    x-min: -0.1,
    name: "plot",
    {
      plot.add(t.zip(y), style: (stroke: none), mark: "o", mark-style: (fill: blue), mark-size: 0.2)
      plot.add(
        domain: (0, 5),
        x => 2.3781818181818135 + 2.4924242424242453 * x - 0.22121212121212158 * x * x,
        label: [$?$],
        style: (stroke: blue)
      )
    }
  )
}),
caption: [Data fiting problem. The objective is to find a smooth curve that fits the data the best.]
)


We consider parameterizing the latent function as a linear combination of a set of basis functions. To be specific, we assume the latent function is a low order polynomial to ensure it is smooth:
$ y = c_0 + c_1 t + c_2 t^2. $

To measure the quality of the fit, we use the mean squares error as the cost function:
$
cal(L)(c_0, c_1, c_2) = sum_(i=1)^n (y_i - (c_0 + c_1 t_i + c_2 t_i^2))^2.
$
The objective is to find the coefficients $c_0, c_1, c_2$ that minimizes the mean squares error.
The more basis functions we use, the more likely to have a lower error. But, it's also more likely to *overfit* the data.
In matrix form, the cost function can be written as:
$ cal(L)(x) = norm(A x - b)_2 $
where
$
A = mat(1, t_1, t_1^2; 1, t_2, t_2^2; dots.v, dots.v, dots.v; 1, t_n, t_n^2), quad
x = vec(c_0, c_1, c_2), quad
b = vec(y_1, y_2, dots.v, y_n).
$

It is a standard least squares problem, and can be solved using QR decomposition:
```julia
julia> using LinearAlgebra

julia> t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
julia> y = [2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4, 9.0];
julia> A = hcat(ones(length(t)), t, t.^2); # `hcat` is the horizontal concatenation
julia> x = (A' * A) \ (A' * b)    # `'` is the Hermitian conjugate
3-element Vector{Float64}:
  2.3781818181818135
  2.4924242424242453
 -0.22121212121212158
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

== Fast Fourier Transform

The Fourier transform is a linear transformation to a function that widely used in signal process, image processing and physics. For a complex-valued function $f(x)$, the Fourier transform is defined as:

$ g(u) = cal(F)(f(x)) = integral_(-infinity)^infinity e^(-2 pi i u x) f(x) dif x $

Here, $u$ represents frequency in the momentum space, while $x$ represents position in the physical space. The inverse Fourier transform recovers the original function:

$ f(x) = cal(F)^(-1)(g(u)) = 1/(2pi)integral_(-infinity)^infinity e^(2 pi i u x) g(u) dif u $

Let us consider the descrete case, where a function is represented as a vector.
Let $x$ be a vector of length $n$, then the Fourier transformation on $x$ is defined as
$ y_i=sum_(n=0)^(n-1)x_j e^(-(i 2 pi\/n) i j) $

Since this transformation is linear, we can represent it as a matrix

$ F_n = mat(
1 , 1 , 1 , dots , 1;
1 , omega , omega^2 , dots , omega^(n-1);
1 , omega^2 , omega^4 , dots , omega^(2n-2);
dots.v , dots.v , dots.v , dots.down , dots.v;
1 , omega^(n-1) , omega^(2n-2) , dots , omega^((n-1)^2)
) $

where $omega = e^(-2 pi i\/n)$.  This matrix $F_n$ is called the DFT matrix. The inverse transformation is defined as $F_n^dagger x\/n$, i.e. $F_n F_n^dagger = I$.

```julia
using Test, LinearAlgebra

function dft_matrix(n::Int)
    ω = exp(-2π*im/n)
    return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
end
```

```julia
n = 3
Fn = dft_matrix(n)
@test Fn * Fn' ≈ I(n)
```

== The Cooley-Tukey's Fast Fourier transformation (FFT)

The Fast Fourier Transform, developed by Cooley and Tukey, provides an efficient algorithm for computing the Discrete Fourier Transform. The key insight is to recursively divide the problem into smaller subproblems, leading to a significant reduction in computational complexity from $O(n^2)$ to $O(n log n)$.

The algorithm works by decomposing the DFT matrix as:

$ F_n x = mat(
  I_(n/2), D_(n/2);
  I_(n/2), -D_(n/2)
) mat(
  F_(n/2), 0;
  0, F_(n/2)
) vec(x_("odd"), x_("even")) $

where $D_n = "diag"(1, omega, omega^2, ..., omega^(n-1))$ and $omega = e^(-2pi i/n)$

#let quiz(body) = {
  block(
    fill: rgb("#f6f6f6"),
    inset: 1em,
    radius: 4pt,
    [*Quiz:* #body]
  )
}

#quiz[
  What is the computational complexity of evaluating $F_n x$? 
  
  Hint: $T(n) = 2 T(n/2) + O(n)$.
]

```julia
using SparseArrays

@testset "fft decomposition" begin
    n = 4
    Fn = dft_matrix(n)
    F2n = dft_matrix(2n)

    # the permutation matrix to permute elements at 1:2:n (odd) to 1:n÷2 (top half)
    pm = sparse([iseven(j) ? (j÷2+n) : (j+1)÷2 for j=1:2n], 1:2n, ones(2n), 2n, 2n)

    # construct the D matrix
    ω = exp(-π*im/n)
    d1 = Diagonal([ω^(i-1) for i=1:n])

    # construct F_{2n} from F_n
    F2n_ = [Fn d1 * Fn; Fn -d1 * Fn]
    @test F2n * pm' ≈ F2n_
end
```

We implement the $O(n log(n))$ time Cooley-Tukey FFT algorithm.

```julia
function fft!(x::AbstractVector{T}) where T
    N = length(x)
    @inbounds if N <= 1
        return x
    end
 
    # divide
    odd  = x[1:2:N]
    even = x[2:2:N]
 
    # conquer
    fft!(odd)
    fft!(even)
 
    # combine
    @inbounds for i=1:N÷2
       t = exp(T(-2im*π*(i-1)/N)) * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return x
end
```

```julia
@testset "fft" begin
    x = randn(ComplexF64, 8)
    @test fft!(copy(x)) ≈ dft_matrix(8) * x
end
```

The Julia package `FFTW.jl` contains a superfast FFT implementation.

```julia
using FFTW

@testset "fft" begin
    x = randn(ComplexF64, 8)
    @test FFTW.fft(copy(x)) ≈ dft_matrix(8) * x
end
```

== Application 1: Fast polynomial multiplication

Given two polynomials $p(x)$ and $q(x)$:

$ p(x) = sum_(k=0)^(n-1) a_k x^k $
$ q(x) = sum_(k=0)^(n-1) b_k x^k $

Their multiplication is defined as:

$ p(x)q(x) = sum_(k=0)^(2n-2) c_k x^k $

Fourier transformation enables computing this product in $O(n log n)$ time, significantly faster than the naive $O(n^2)$ algorithm.

#block(
  fill: rgb("#f6f6f6"),
  inset: 1em,
  radius: 4pt,
  [
    *Algorithm: Fast Polynomial Multiplication*

    1. Evaluate $p(x)$ and $q(x)$ at $2n$ points $omega^0, dots, omega^(2n-1)$ using DFT. Time: $O(n log n)$

    2. Compute pointwise multiplication:
       $ (p compose q)(omega^j) = p(omega^j) q(omega^j) $ for $j = 0, dots, 2n-1$
       Time: $O(n)$

    3. Interpolate $p compose q$ using inverse DFT to obtain coefficients $c_0, c_1, dots, c_(2n-2)$. Time: $O(n log n)$
  ]
)

This algorithm can also compute vector convolutions. For vectors $a = (a_0, dots, a_(n-1))$ and $b = (b_0, dots, b_(n-1))$, their convolution $c = (c_0, dots, c_(n-1))$ is:

$ c_j = sum_(k=0)^j a_k b_(j-k), quad j = 0,dots,n-1 $

In the following example, we use the `Polynomials` package to define the polynomial and use the FFT algorithm to compute the product of two polynomials.

```julia
using Polynomials
p = Polynomial([1, 3, 2, 5, 6])
q = Polynomial([3, 1, 6, 2, 2])
```

Step 1: evaluate $p(x)$ at $2n-1$ different points.

```julia
pvals = fft(vcat(p.coeffs, zeros(4)))
```

which is equivalent to computing:

```julia
n = 5
ω = exp(-2π*im/(2n-1))
map(k->p(ω^k), 0:(2n-1))
```

The same for $q(x)$.

```julia
qvals = fft(vcat(q.coeffs, zeros(4)))
```

Step 2: Compute $p(x) q(x)$ at $2n-1$ points.

```julia
pqvals = pvals .* qvals
ifft(pqvals)
```

Summarize:

```julia
function fast_polymul(p::AbstractVector, q::AbstractVector)
    pvals = fft(vcat(p, zeros(length(q)-1)))
    qvals = fft(vcat(q, zeros(length(p)-1)))
    pqvals = pvals .* qvals
    return real.(ifft(pqvals))
end

function fast_polymul(p::Polynomial, q::Polynomial)
    Polynomial(fast_polymul(p.coeffs, q.coeffs))
end
```

A similar algorithm has already been implemented in package `Polynomials`. One can easily verify the correctness.

```julia
p * q
fast_polymul(p, q)
```

== Application 2: Image compression

The Fourier transform is particularly effective for image compression. Consider a grayscale image represented as a matrix of pixel values. The process works as follows:

1. Convert the image to frequency domain using 2D FFT
2. Most of the frequency components will be close to zero
3. We can discard small coefficients (below a threshold) with minimal visual impact
4. Store the remaining coefficients in sparse format
5. Recover the image using inverse FFT

The compression ratio depends on the image content and the chosen threshold. For many natural images, compression ratios of 10:1 or better are achievable while maintaining good visual quality.

#figure(
  caption: "FFT-based image compression example showing original, frequency domain representation, and reconstructed image"
)[
  // Note: Replace with actual image grid showing the compression process
  #grid(
    columns: 3,
    gutter: 1em,
    [Original],
    [Frequency Domain],
    [Reconstructed]
  )
]

The effectiveness of FFT-based compression demonstrates why Fourier transforms are fundamental to modern image and signal processing applications, including JPEG compression.

== Implementation and Practical Considerations

=== Efficient Implementation

The FFT algorithm can be implemented efficiently using various approaches. Here's a pseudocode representation of the Cooley-Tukey algorithm:

```python
function fft(x):
    N = length(x)
    if N <= 1:
        return x
    
    // Divide
    even = x[0:N:2]
    odd = x[1:N:2]
    
    // Conquer
    even = fft(even)
    odd = fft(odd)
    
    // Combine
    for k in 0..<N/2:
        t = exp(-2πi*k/N) * odd[k]
        x[k] = even[k] + t
        x[k + N/2] = even[k] - t
    
    return x
```

=== Performance Considerations

When implementing FFT, several factors affect performance:

1. *Memory Access Patterns*: The algorithm's efficiency depends heavily on cache utilization. In-place implementations can reduce memory usage.

2. *Vector Operations*: Modern processors support SIMD (Single Instruction Multiple Data) operations, which can significantly accelerate complex arithmetic.

3. *Input Size*: FFT is most efficient when the input size is a power of 2. For other sizes, zero-padding or more complex algorithms may be needed.

=== Common Pitfalls

#block(
  fill: rgb("#f8f0e8"),
  inset: 1em,
  radius: 4pt,
  [
    *Common Implementation Issues:*
    
    - Numerical stability in floating-point arithmetic
    - Incorrect handling of array indices
    - Inefficient memory allocation
    - Poor cache utilization in large transforms
  ]
)

=== Advanced Topics

Several variations of FFT exist for specific use cases:

- *Real FFT*: Optimized for real-valued inputs
- *Parallel FFT*: Designed for multi-core or distributed systems
- *Short FFT*: Specialized for small transform sizes
- *Multi-dimensional FFT*: Efficient transforms for higher-dimensional data

=== Further Reading

For deeper understanding of FFT implementations and optimizations, consider:

- "Numerical Recipes" by Press et al.
- "The Fast Fourier Transform and Its Applications" by Brigham
- FFTW documentation (www.fftw.org)

#quiz[
  How would you modify the basic FFT algorithm to handle real-valued inputs more efficiently?
  
  Hint: Consider the symmetry properties of the Fourier transform of real signals.
]
#bibliography("refs.bib")