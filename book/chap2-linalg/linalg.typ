#import "../book.typ": book-page
#import "@preview/cetz:0.2.2": *

#show: book-page.with(title: "Matrix Computation")
#let exampleblock(it) = block(fill: rgb("#ffffff"), inset: 1em, radius: 4pt, stroke: black, it)
#set math.equation(numbering: "(1)")

= Matrix Computation

In this chapter, we explore fundamental operations in matrix computation, including matrix multiplication, linear systems, matrix decompositions, and eigenvalue problems. We'll provide practical examples and applications for each concept. For a comprehensive reference on matrix computations, see @Golub2016.

= Linear Algebra Basics
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
A x = b.
$
A straightforward way to solve this is to compute the inverse of $A$ and then multiply it by $b$:
$
x = A^(-1) b.
$
However, computing the inverse of a matrix is computationally expensive and numerically unstable. A more efficient and stable way to solve the linear system is to use the LU decomposition. The LU decomposition of a matrix $A in RR^(n times n)$ is a factorization of the form
$
A = L U
$
where $L$ is a lower triangular matrix, and $U$ is an upper triangular matrix.
Solving a triangular system is much more efficient than solving a general linear system, which can be done by _forward and backward substitution_ in $O(n^2)$ time.
Given a linear system $A x = b$, we can reformulate it as $L U x = b$, and solve it by first solving $L y = b$ and then solving $U x = y$.
The LU decomposition can be unstable when the diagonal elements of $A$ are small. To improve the stability, we can use the LU decomposition with _pivoting_, which is a variant of the LU decomposition that allows us to swap rows or columns or both of $A$ to ensure numerical stability. By default, Julia's `lu` function performs partial pivoting, which swaps rows of $A$ to ensure numerical stability:
$ P A = L U. $
where $P$ is a permutation matrix for pivoting rows.

#exampleblock([
    *Example: Solving a linear system*

    Let us consider the following system of linear equations
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
    julia> A = [2 3 -2; 3 2 3; 4 -3 2]
    julia> b = [1, 2, 3]
    julia> x = A \ b
    3-element Vector{Float64}:
      0.6666666666666666
     -0.07692307692307693
      0.05128205128205128
    julia> A * x  # should be close to b
    ```
    The "`\`" operator effectively performs the following steps:
    ```julia
    julia> using LinearAlgebra
    julia> res = lu(A)  # lu decomposition, `res.L * res.U ≈ res.P * A`
    julia> y = LowerTriangular(res.L) \ (res.P*b)  # forward substitution
    julia> x = UpperTriangular(res.U) \ y  # backward substitution
    3-element Vector{Float64}:
      0.6666666666666666
     -0.07692307692307693
      0.05128205128205128
    ```
    Here, in order to ensure that Julia recognizes `res.L` and `res.U` as triangular matrices, we need to wrap them with `LowerTriangular(res.L)` and `UpperTriangular(res.U)`.
  ]
)


== Cholesky Decomposition
Cholesky decomposition is a variant of LU decomposition for _symmetric positive-definite_ matrices. For a symmetric positive-definite matrix $A in CC^(n times n)$, $A succ.eq 0 $, the Cholesky decomposition provides a factorization:
$ A = L L^dagger $
where $L in CC^(n times n)$ is lower triangular. Cholesky decomposition is often used as a compact representation of a positive-definite matrix. In Julia, the `cholesky` function returns a `Cholesky` object, which contains the lower triangular matrix $L$ and the determinant of $A$.
```julia
julia> A = [2 1; 1 3]  # symmetric positive-definite

julia> C = cholesky(A)
Cholesky{Float64, Matrix{Float64}}
U factor:
2×2 UpperTriangular{Float64, Matrix{Float64}}:
 1.41421  0.707107
  ⋅       1.58114

julia> C.L * C.L' ≈ A  # output: true
```

== Least Squares Problem and QR Decomposition

The least squares problem is to find a vector $x in RR^n$ that minimizes the residual
$
min_x norm(A x - b)_2
$ <eq:lsq-problem>
where $A in RR^(m times n)$ and $b in RR^m$.

A linear equation is just a special case of the least squares problem, where the *residual* is zero.
The least squares problem "makes sense" only when $A$ is *over-determined* (meaning having too many equations such that not all can be satisfied), i.e. $m > n$.

Converting the least squares problem to a normal equation is a straightforward approach to solve it.
We first square and expand the residual in @eq:lsq-problem as $(A x - b)^T (A x - b).$
The minimum is attained when the gradient of the quadratic function is zero, i.e.
$
nabla_(x) (x^T A^T A x - 2 x^T A^T b + b^T b) = 2 A^T A x - 2 A^T b = 0\
arrow.double.r x = (A^T A)^(-1) A^T b,
$
which is the _normal equation_.
Directly solving the normal equation with matrix inversion is numerically unstable. This is due to the fact that the condition number of $A^T A$ is the square of the condition number of $A$. The relation between the condition numbers and the numerical stability is discussed in a later section.

The QR decomposition of a matrix provides a more stable way to solve the least squares problem. Given a matrix $A in bb(C)^(m times n)$, the QR decomposition is a factorization of the form
$
A = Q R
$
where $Q in bb(C)^(m times min(m, n))$ is an orthogonal matrix (i.e. $Q^dagger Q = I$) and $R in bb(C)^(min(m, n) times n)$ is an upper triangular matrix.
Let $A = Q R$, the least squares problem $min_x ||A x - b||_2^2$ is equivalent to
$
  min_x ||Q R x - b||_2^2 = underbrace(min_y ||R x - Q^dagger b||_2^2, "zero") + ||Q^dagger_bot b||_2^2\
  arrow.double.r R x = Q^dagger b
$
where $Q^dagger_bot$ is the orthogonal complement of $Q^dagger$, i.e. $Q^dagger_bot Q = 0$ and $Q^dagger_bot Q^dagger = I$. For a unitary matrix $Q$, we have $||Q x||_2 = ||x||_2$. However, this is not true for $Q^dagger$, i.e. $||Q^dagger x||_2 <= ||x||_2$, where the equality holds if and only if $x$ is in the column space of $Q$. This explains why we have an extra term $||Q^dagger_bot b||_2^2$ in the above equation.

To summarize, to solve a least squares problem, we first compute the QR decomposition of $A = Q R$, then solve an upper triangular system $R x = Q^dagger b$ in $O(n^2)$ time. The condition number of $R$ is the same as that of $A$, hence the solution is numerically stable.

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

It is a standard least squares problem, and we first solve it using the normal equation:
```julia
julia> using LinearAlgebra

julia> t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
julia> y = [2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4, 9.0];
julia> A = hcat(ones(length(t)), t, t.^2); # `hcat` is the horizontal concatenation

julia> x = (A' * A) \ (A' * y)    # `'` is the Hermitian conjugate
3-element Vector{Float64}:
  2.3781818181818135
  2.4924242424242453
 -0.22121212121212158
```

Then, we compute the QR decomposition of $A$:
```julia
julia> Q, R = qr(A)
LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
Q factor: 10×10 LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}
R factor:
3×3 Matrix{Float64}:
 -3.16228  -7.11512  -22.5312
  0.0       4.54148   20.4366
  0.0       0.0        5.74456

julia> Q' * Q    # Identity matrix
julia> Q * Q'    # Not identity matrix
julia> rank(Q * Q')   # not full rank

julia> x = R \ (Matrix(Q)' * y)  # solve Rx = Q'y
 3-element Vector{Float64}:
  2.3781818181818197
  2.492424242424242
 -0.22121212121212133
```
Here, we did not see a big difference in the solution due to the small size of the matrix. But for larger matrices, the QR decomposition is more stable and accurate.
]
)

== Singular value decomposition
The singular values decomposition (SVD) of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = U S V^dagger
$
where $U in bb(C)^(m times m)$ and $V in bb(C)^(n times n)$ are unitary matrices (i.e. $U^dagger U = I$ and $V^dagger V = I$), and $S = "diag"(lambda_1, lambda_2, dots, lambda_n)$ is a diagonal matrix with *non-negative* real numbers on the diagonal.

- _Remark_: the SVD is a generalization of the eigendecomposition of a matrix. The diagonal elements of $S$ are the singular values arranged in descending order.
- _Remark_: For real matrices, $U$ and $V$ are orthogonal matrices (i.e. $U^T U = I$ and $V^T V = I$).

The SVD generalizes eigenvalue decomposition to non-square matrices and provides a powerful tool for matrix analysis, dimensionality reduction, and solving least squares problems. In Julia, we compute SVD using the `svd` function:

```julia
julia> A = [1 2; 3 4; 5 6]
julia> F = svd(A)
SVD{Float64, Float64, Matrix{Float64}, Vector{Float64}}
U factor:
3×2 Matrix{Float64}:
 -0.229848   0.883461
 -0.524745   0.240782
 -0.819642  -0.401896
singular values:
2-element Vector{Float64}:
 9.525518091565107
 0.514300580658644
Vt factor:
2×2 Matrix{Float64}:
 -0.619629  -0.784894
 -0.784894   0.619629
julia> F.U * Diagonal(F.S) * F.Vt ≈ A  # output: true
```


== Eigen-decomposition
The eigenvalues and (right) eigenvectors of a matrix $A in bb(C)^(n times n)$ are the solutions to the equation
$
A x = lambda x
$
where $lambda$ is a scalar and $x$ is a non-zero vector. The eigen-decomposition of a matrix is a factorization of the form
$
A = P D P^(-1)
$
where $D$ is a diagonal matrix and the diagonal elements of $D = "diag"(lambda_1, lambda_2, dots, lambda_n)$ are the eigenvalues of $A$. The $i$-th column of $P$ is the right eigenvectors of $A$ that corresponds to the eigenvalue $lambda_i$. For symmetric matrices, the eigenvalues are real and eigenvectors are orthogonal, i.e. $P^dagger P = I$.

Eigen-decomposition is useful for computing _matrix functions_. Consider an analytic function $f$ defined by a power series and matrix $A in CC^(n times n)$:
$ f(A) = sum_(i=0)^infinity a_i A^i, $
its value can be computed by first diagonalizing $A = P D P^(-1)$, then evaluate $f(A)$ as $P f(D) P^(-1)$. The function $f$ applied to a diagonal matrix is element-wise as $f(D) = "diag"(f(lambda_1), f(lambda_2), dots, f(lambda_n)).$

#exampleblock[
    *Example: Matrix Exponential*
    
    Compute the matrix function $exp(A)$ for
    $ A = mat(1, 2; 3, 4). $

    The first approach is using Julia's `exp` function:
    ```julia
    julia> A = [1 2; 3 4]
    julia> exp(A)
    2×2 Matrix{Float64}:
      51.969   74.7366
     112.105  164.074
    ```

    Alternatively, we can use the eigen-decomposition:
    ```julia
    julia> using LinearAlgebra
    julia> D, P = eigen(A)   # returns eigenvalues and eigenvectors
    Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
    values:
    2-element Vector{Float64}:
     -0.3722813232690143
      5.372281323269014
    vectors:
    2×2 Matrix{Float64}:
     -0.824565  -0.415974
      0.565767  -0.909377
    julia> P * Diagonal(exp.(D)) * inv(P)
    2×2 Matrix{Float64}:
      51.969   74.7366
     112.105  164.074
    ```
  ]

=== Application: eigenmodes of a vibrating spring chain

#figure(canvas(length: 0.8cm, {
  import draw: *
  let (dx, dy) = (2.0, 1.0)
  let s(it) = text(12pt)[#it]
  let u = (0, 0, 0, -0.6, 0.8, 0, 0, 0)
  for i in range(8){
    circle((i * dx + u.at(i), 0), radius: 0.2, name: "atom" + str(i))
  }
  for i in range(7){
    decorations.wave(line("atom" + str(i), "atom" + str(i + 1)), amplitude: 0.2)
  }
  line((3 * dx, 1), (3 * dx, 0), mark: (end: "straight"))
  line((4 * dx, 1), (4 * dx, 0), mark: (end: "straight"))
  content((3.5 * dx, 1.8), s[equilibrium position])

  line((3 * dx, -0.3), (3 * dx + u.at(3), -0.3), mark: (end: "straight"), name: "d1")
  content((rel: (0, -0.3), to: "d1.mid"), s[$u_4$])
  line((4 * dx, -0.3), (4 * dx + u.at(4), -0.3), mark: (end: "straight"), name: "d2")
  content((rel: (0, -0.3), to: "d2.mid"), s[$u_5$])

  //content((1.5 * dx, 0.8), s[$c/2 (u_i - u_(i+1))^2$])
}),
)

Let us consider a system with $n$ atoms connected by springs. The potential energy of the spring system has the following quadratic form:
$
V(bold(u)) = c/2 sum_(i=1)^(n-1) (u_i - u_(i+1))^2
$
where $c$ is the stiffness, and $u_i$ is the displacement of the $i$-th atom. The end atoms are fixed, so we have $u_0 = u_(n+1) = 0$. Its dynamics can be described by the Newton's second law
$
m dot.double(u)_i = c (u_(i+1) - u_i) - c (u_i - u_(i-1))
$ <eq:spring-dynamics>
where $m$ is the mass of the atom.

It is known that a spring system has the eigenmodes of the form
$
u_i (t) = A_i cos(omega t + phi_i)
$ <eq:spring-eigenmodes>
where $A_i$ is the amplitude, $omega$ is the eigenfrequency, and $phi_i$ is the phase of the $i$th atom. All the atoms oscillate with the same eigenfrequency $omega$ around their equilibrium positions. In the following, we will derive the eigenfrequencies and eigenmodes of the spring system.

By inserting @eq:spring-eigenmodes into @eq:spring-dynamics, we have
$
-m omega^2 u_i = c (u_(i+1) - u_i) - c (u_i - u_(i-1))
$
Then finding the eigenmodes of the spring system is equivalent to solving the following eigenvalue problem:
$
C vec(u_1, u_2, dots.v, u_n) = -M omega^2 vec(u_1, u_2, dots.v, u_n)
$
where $
        M = "diag"(m_1, m_2, dots, m_n), quad
        C = mat(-c, c, 0, dots, 0, 0; c, -2c, c, dots, 0, 0; dots.v, dots.v, dots.v, dots.down, dots.v, dots.v; 0, 0, 0, dots, 0, c; 0, 0, 0, dots, c, -c).
      $

In the following, we will solve the eigenmodes problem for a 5-atom spring system with $m = c = 1.0$.
```julia
julia> M = C = 1.0;
julia> C_matrix = [-C C 0 0 0; C -2C C 0 0; 0 C -2C C 0; 0 0 C -2C C; 0 0 0 C -C];
julia> evals, evecs = LinearAlgebra.eigen(C_matrix);
julia> second_omega = sqrt(-evals[2]/M)  # the second eigenfrequency
1.618033988749894

julia> second_mode = evecs[:, 2]  # the second eigenmode, each element is the amplitude of the $i$-th atom
5-element Vector{Float64}:
  0.37174803446018484
 -0.6015009550075462
  1.4023804401251382e-15
  0.601500955007545
 -0.3717480344601845
```

```julia
julia> u(t) = second_mode .* cos.(-second_omega .* t) # (ϕi=0)
u (generic function with 1 method)

julia> u(1.0)  # atom locations offsets at t=1.0
5-element Vector{Float64}:
 -0.017553977969578697
  0.028402932992545194
 -6.622053936793937e-17
 -0.028402932992545135
  0.01755397796957868
```

#figure(image("images/springs-demo.gif", width: 300pt), caption: [One of the eigenmodes of a spring chain. The simulation result is obtained by solving the differential equation with the Verlet algorithm. Exact result is given by the eigen-decomposition of the stiffness matrix.])


Any initial condition can be expressed as a linear combination of these eigenmodes. For implementation details, see the #link("https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/SpringSystem")[source code].

== Fast Fourier Transform

The Fourier transform is a *linear transformation* that widely used in signal process, image processing and physics.
It can transform a continuous function to peak-like functions in the _frequency domain_. For a complex-valued function $f(x)$, the Fourier transform and its inverse transform are defined as:

$ g(u) = cal(F)(f(x)) = integral_(-infinity)^infinity e^(-2 pi i u x) f(x) diff x\
f(x) = cal(F)^(-1)(g(u)) = 1/(2pi)integral_(-infinity)^infinity e^(2 pi i u x) g(u) diff u
$
Here, $u$ represents frequency in the _momentum space_, while $x$ represents position in the _physical space_.

When limiting the function to a finite domain, the Fourier transform can be computed by a discrete Fourier transform (DFT). Let $bold(x) = (x_0, x_1, dots, x_(n-1))$ be a vector of length $n$, then the Fourier transformation on $bold(x)$ is defined as
$ y_i = sum_(j=0)^(n-1) x_j e^(-(i 2 pi\/n) i j). $

Since any linear transformation can be represented as a matrix, the DFT can be represented as a matrix:

$
bold(y) = F_n bold(x),\
F_n = mat(
1 , 1 , 1 , dots , 1;
1 , omega , omega^2 , dots , omega^(n-1);
1 , omega^2 , omega^4 , dots , omega^(2n-2);
dots.v , dots.v , dots.v , dots.down , dots.v;
1 , omega^(n-1) , omega^(2n-2) , dots , omega^((n-1)^2)
).
$

where $omega = e^(-2 pi i\/n)$.  This matrix $F_n$ is called the _DFT matrix_, and the inverse transformation is defined as $F_n^dagger x\/n$. We have $F_n F_n^dagger = I$.

DFT is extremely fast to compute. It can be evaluated in $O(n log n)$ time using the Fast Fourier Transform (FFT) algorithm, first proposed by Cooley and Tukey in 1965 @Cooley1965. It does not construct the DFT matrix explicitly, but instead uses a divide-and-conquer strategy.
The Julia package `FFTW.jl` contains an extremely efficient implementation of FFT.

```julia
julia> using FFTW
julia> using LinearAlgebra

julia> function dft_matrix(n::Int)  # the definition of the DFT matrix
           ω = exp(-2π*im/n)
           return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
       end

julia> n = 3
julia> Fn = dft_matrix(n)
3×3 Matrix{ComplexF64}:
 1.0+0.0im   1.0+0.0im        1.0+0.0im
 1.0+0.0im  -0.5-0.866025im  -0.5+0.866025im
 1.0+0.0im  -0.5+0.866025im  -0.5-0.866025im
julia> Fn * (Fn'/3) ≈ I(n)    # output: true

julia> x = ones(n)            # a uniform vector
julia> y = FFTW.fft(copy(x))  # a uniform function is transformed to delta function
3-element Vector{ComplexF64}:
 3.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
julia> y ≈ dft_matrix(n) * x  # output: true
```

== Application 1: Fast polynomial multiplication

Given two polynomials $p(x)$ and $q(x)$:

$ p(x) = sum_(k=0)^(n-1) a_k x^k, quad q(x) = sum_(k=0)^(n-1) b_k x^k $

Their multiplication is defined as:

$ p(x)q(x) = sum_(k=0)^(2n-2) c_k x^k $

Fourier transformation enables computing this product in $O(n log n)$ time, significantly faster than the naive $O(n^2)$ algorithm.

#exampleblock[
*Algorithm: Fast Polynomial Multiplication*

1. Evaluate $p(x)$ and $q(x)$ at $2n$ points $omega^0, dots, omega^(2n-1)$ using DFT.
2. Compute pointwise multiplication:
    $ (p compose q)(omega^j) = p(omega^j) q(omega^j) $ for $j = 0, dots, 2n-1$.
3. Interpolate $p compose q$ using inverse DFT to obtain coefficients $c_0, c_1, dots, c_(2n-2)$.

The first and third steps take $O(n log n)$ time, and the second step takes $O(n)$ time.
]

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
(TBD)

#bibliography("refs.bib")