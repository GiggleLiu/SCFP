#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Matrix computation: Applications and basics],
  subtitle: [],
  author: [Jin-Guo Liu],
  date: datetime.today(),
  institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
)

// Extract methods
#let (init, slides) = utils.methods(m)
#show: init

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(m)
#show: slides.with()

#outline-slide()

= Linear Equations

== Matrix computation
#timecounter(1)

Matrix computations @Golub2013. Cited by 85063 papers!
#figure(image("images/golub.jpg", width: 150pt))

== Notations
#timecounter(1)

- Real Scalar: $x in RR$
- Real Matrix: $A in RR^(m times n)$
- Complex Scalar: $z in CC$
- Transpose: $A^T$
- Complex Conjugate: $A^*$
- Hermitian conjugate: $A^dagger ("or" A^H) = (A^T)^*$

== System of Linear Equations
#timecounter(1)

Let $A in RR^(n times n)$ be an invertible square matrix and $b in RR^n$ be a vector. Solving a linear equation means finding a vector $x in RR^n$ such that
$
A x = b
$

== Example
#timecounter(2)

Let us consider the following system of linear equations
$
2 x_1 + 3 x_2 - 2 x_3 &= 1, \
3 x_1 + 2 x_2 + 3 x_3 &= 2, \
4 x_1 - 3 x_2 + 2 x_3 &= 3.
$
The matrix form of the system is
$
A x = b\
A = mat(2, 3, -2; 3, 2, 3; 4, -3, 2), quad
x = vec(x_1, x_2, x_3), quad
b = vec(1, 2, 3)
$

== Solving a system of linear equations
#timecounter(2)

#box(text(16pt)[```julia
julia> A, b = [2 3 -2; 3 2 3; 4 -3 2], [1, 2, 3];

julia> x = A \ b   # solve A x = b
3-element Vector{Float64}:
  0.6666666666666666
 -0.07692307692307693
  0.05128205128205128

julia> A * x
3-element Vector{Float64}:
 0.9999999999999999
 2.0
 3.0
```])

== Least Squares Problem
#timecounter(1)

The least squares problem is to find a vector $x in RR^n$ that minimizes the residual
$
min_x ||A x - b||_2
$
where $A in RR^(m times n)$ and $b in RR^m$.

- _Remark_: linear equation is a special case of the least squares problem when the *residual* is zero.
- _Remark_: the least squares problem "makes sense" only when $A$ is *over-determined* (meaning having too many equations such that not all can be satisfied), i.e. $m > n$.

== Example: data fitting
#timecounter(1)

Objective: Find a *smooth* curve that fits the data the *best*.
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
)

== Mean squares error
#timecounter(2)
*Guess*: the latent function is a quadratic function $y = c_0 + c_1 t + c_2 t^2$ .

We hope it can minimize the mean squares error:

$
cal(L)(c_0, c_1, c_2) = sum_(i=1)^n (y_i - (c_0 + c_1 t_i + c_2 t_i^2))^2
$

- _Remark_: The latent function can be any linear combination of a set of basis functions. The more basis functions we use, the more likely it can fit the data. But, its also more likely to *overfit* the data.

== Matrix representation
#timecounter(2)
$
min_x ||A x - b||_2^2
$
where
$
A = mat(1, t_1, t_1^2; 1, t_2, t_2^2; dots.v, dots.v, dots.v; 1, t_n, t_n^2), quad
x = vec(c_0, c_1, c_2), quad
b = vec(y_1, y_2, dots.v, y_n)
$

- _Note_: the $p$-norm of a vector is defined as $||x||_p = (sum_(i=1)^n |x_i|^p)^(1/p)$. e.g. 1-norm is the summation of the absolute values of the elements, and $infinity$-norm is the maximum absolute value of the elements.

== Normal equations
#timecounter(2)

Think:
1. we want to minimize $||A x - b||_2^2$ w.r.t. $x$.
2. which is equivalent to minimizing $(A x - b)^T (A x - b) = x^T A^T A x - 2 x^T A^T b + b^T b$
3. the minimum is attained when the gradient of the quadratic function is zero, i.e.
$
nabla_(x) (x^T A^T A x - 2 x^T A^T b + b^T b) = 2 A^T A x - 2 A^T b = 0
$
4. Finally, we get the normal equation:
$
x = (A^T A)^(-1) A^T b
$

== Live coding: solving the least squares problem with normal equations
#timecounter(2)

#box(text(16pt)[```julia
julia> using LinearAlgebra

julia> t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
julia> y = [2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4, 9.0];
julia> A = hcat(ones(length(t)), t, t.^2);
julia> x = (A' * A) \ (A' * b)
3-element Vector{Float64}:
  2.3781818181818135
  2.4924242424242453
 -0.22121212121212158
```])

- _Note_: "`'`" is the Hermitian conjugate of a matrix. For real matrices, it is the transpose.
- _Note_: "`hcat`" is the horizontal concatenation of matrices.

== Stability of the normal equation method
#timecounter(1)
- _Fact_: the normal equation method is numerically *unstable*. Why?

TODO: add an example to show the instability.

Short answer:
- #link("https://en.wikipedia.org/wiki/Floating-point_arithmetic", "Floating-point numbers") for storing real numbers has limited precision.
- We use the _condition number_ $kappa(A)$ to measure the instability of a linear system.
- $A^T A$ squares the condition number of $A$.



= Errors and condition number

== Floating-point numbers and relative errors
#timecounter(2)

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
#timecounter(2)

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
#timecounter(2)
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
#timecounter(2)

- _Condition number_ of a matrix $A$ is defined as $kappa(A) = ||A|| ||A^(-1)|| >=1$. If the condition number is close to 1, the matrix is _well-conditioned_, otherwise it is _ill-conditioned_.

- _Remark_: there are two popular norms for matrices: the Frobenius norm and the $p$-norms. Here, we use the $p=2$ norm for simplicity.
  - Frobenius norm: $||A||_F = sqrt(sum_(i j) |a_(i j)|^2)$
  - $p$-norm: $||A||_p = max_(x != 0) (||A x||_p)/ (||x||_p)$, for $p = 2$, it is the spectral norm $||A||_2 = sigma_1(A)$, the largest _singular value_ of $A$.

== Meaning of the condition number
#timecounter(2)

- _Remark_: meaning of the condition number: if we solve the linear system $A x = b$ with a small perturbation $b + delta b$, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

$
  A(x + delta x) = b + delta b\
  arrow.double.r (||delta x||) / (||x||) = (||A^(-1) delta b||) / (||A^(-1) b||) <= (lambda_1(A^(-1))) / (lambda_n (A^(-1))) (||delta b||) / (||b||)
$
where $lambda_1(A^(-1))$ and $lambda_n (A^(-1))$ ($= lambda_1(A)^(-1)$) are the largest and smallest _singular values_ of $A^(-1)$, respectively.

Hence, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

== Singular values decomposition
#timecounter(2)
The singular values decomposition (SVD) of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = U S V^dagger
$
where $U in bb(C)^(m times m)$ and $V in bb(C)^(n times n)$ are unitary matrices (i.e. $U^dagger U = I$ and $V^dagger V = I$), and $S = "diag"(lambda_1, lambda_2, dots, lambda_n)$ is a diagonal matrix with *non-negative* real numbers on the diagonal.

- _Remark_: the SVD is a generalization of the eigendecomposition of a matrix. The diagonal elements of $S$ are the singular values arranged in descending order.
- _Remark_: For real matrices, $U$ and $V$ are orthogonal matrices (i.e. $U^T U = I$ and $V^T V = I$).

== SVD and condition number
#timecounter(2)

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
#timecounter(2)

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

= QR decomposition
== Stabilize the normal equation: QR Decomposition
#timecounter(2)

The QR decomposition of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = Q R
$
where $Q in bb(C)^(m times min(m, n))$ is an orthogonal matrix (i.e. $Q^dagger Q = I$) and $R in bb(C)^(min(m, n) times n)$ is an upper triangular matrix.

== Solving linear systems with QR decomposition
#timecounter(2)
Let $A = Q R$, the least squares problem $min_x ||A x - b||_2^2$ is equivalent to
$
  min_x ||Q R x - b||_2^2 = underbrace(min_y ||R x - Q^dagger b||_2^2, "zero") + ||Q^dagger_bot b||_2^2\
  arrow.double.r R x = Q^dagger b
$
where $Q^dagger_bot$ is the orthogonal complement of $Q^dagger$, i.e. $Q^dagger_bot Q = 0$ and $Q^dagger_bot Q^dagger = I$.

- _Remark_: For a unitary matrix $Q$, $||Q x||_2 = ||x||_2$. However, $||Q^dagger x||_2 <= ||x||_2$, where the equality holds if and only if $x$ is in the column space of $Q$.
- _Remark_: For an upper triangular matrix $R$, the solution of $R x = y$ can be found by _backward substitution_ in $O(n^2)$ time. $kappa(R) = kappa(A)$.

== Live coding: solving the least squares problem with QR decomposition
#timecounter(2)
#box(text(16pt)[```julia
julia> Q, R = qr(A)

julia> Q' * Q    # Identity matrix
julia> Q * Q'
julia> rank(Q * Q')

julia> x = R \ (Matrix(Q)' * y)
```])

== Alternative approach: LU Decomposition
#timecounter(2)

The LU decomposition of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = L U
$
where $L$ is a lower triangular matrix, and $U$ is an upper triangular matrix.

- _Remark_: Given a linear system $A x = b$, we can reformulate it as $L U x = b$, and solve it by first solving $L y = b$ and then solving $U x = y$.

== Live coding: LU Decomposition
#timecounter(2)

#box(text(16pt)[```julia
julia> using LinearAlgebra

julia> lures = lu(A, NoPivot())  # pivot rows by default

julia> lures.L * lures.U ≈ A

julia> forward = LowerTriangular(lures.L) \ (lures.P * b)
julia> x = UpperTriangular(lures.U) \ forward
```])

- _Remark_: pivoting can improve the stability of the LU decomposition.
- _Remark_: for symmetric _positive definite matrices_ (i.e. $A = A^T$ and $x^T A x > 0$ for any $x != 0$), we have the Cholesky decomposition $A = L L^dagger$, which is a special case of the LU decomposition.

= Eigenvalues and eigenvectors
== Eigen-decomposition
The eigenvalues and eigenvectors of a matrix $A in bb(C)^(n times n)$ are the solutions to the equation
$
A x = lambda x
$
where $lambda$ is a scalar and $x$ is a non-zero vector.

In Julia, we can find the eigenvalues and eigenvectors of a matrix using the `eigen` function.

```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4
```

```julia
julia> eigen(A)
LinearAlgebra.Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
values:
2-element Vector{Float64}:
 -0.3722813232690143
  5.372281323269014
vectors:
2×2 Matrix{Float64}:
 -0.824565  -0.415974
  0.565767  -0.909377
```

== Example: eigenmodes of a vibrating string (or atomic chain)
This example is about solving the dynamics of a vibrating string.

![](image-2.png)

[Image source and main reference](https://lampz.tugraz.at/~hadley/ss1/phonons/1d/1dphonons.php)

== Dynamics of a vibrating string

The dynamics of a one dimensional vibrating string can be described by the Newton's second law
$
M dot.double(u) = C(u_{i+1} - u_i) - C(u_i - u_{i-1})
$
where $M$ is the mass matrix, $C$ is the stiffness, and $u_i$ is the displacement of the $i$th atom. The end atoms are fixed, so we have $u_0 = u_{n+1} = 0$.

== Eigen-decomposition of the mass matrix
We assume all atoms have the same eigenfrequency $omega$ and the displacement of the $i$th atom is given by
$
u_i(t) = A_i cos(omega t + phi_i)
$
where $phi_i$ is the phase of the $i$th atom.

== Eigenmodes of the vibrating string

$
mat(-C, C, 0, dots, 0; C, -2C, C, dots, 0; 0, C, -2C, dots, 0; dots.v, dots.v, dots.v, dots.down, dots.v; 0, 0, 0, dots, -C)

vec(A_1, A_2, A_3, dots.v, A_n)
= -omega^2 M vec(A_1, A_2, A_3, dots.v, A_n)
$
The eigenvalues $omega^2$ are the squared eigenfrequencies.

== 5-atom vibrating string
```julia
julia> M = C = 1.0;

julia> C_matrix = [-C C 0 0 0; C -2C C 0 0; 0 C -2C C 0; 0 0 C -2C C; 0 0 0 C -C];

julia> evals, evecs = LinearAlgebra.eigen(C_matrix);
```

```julia
julia> second_omega = sqrt(-evals[2]/M)
1.618033988749894

julia> second_mode = evecs[:, 2]
5-element Vector{Float64}:
  0.37174803446018484
 -0.6015009550075462
  1.4023804401251382e-15
  0.601500955007545
 -0.3717480344601845
```

== Example: eigenmodes of a vibrating string

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

![](springs-demo.gif)

== Matrix functions

Suppose we have a matrix $A in bb(C)^(n times n)$ and an analytic function $f$ defined with a power series

$
f(A) = sum_(i=0)^infinity a_i A^i.
$

== Matrix functions

1. Diagonalize the matrix $A$ as $A = P D P^(-1)$, where $D$ is a diagonal matrix and $P$ is a matrix whose columns are the eigenvectors of $A$.
2. Compute the matrix function $f(A)$ as $f(A) = P f(D) P^(-1)$.
3. Compute the matrix function $f(D)$ by applying the function $f$ to the diagonal elements of $D$.
4. Compute the matrix function $f(A)$ by multiplying the matrices $P$, $f(D)$, and $P^(-1)$, i.e. $f(A) = P f(D) P^(-1)$.

== Example: Matrix exponential

```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4


julia> exp(A)
2×2 Matrix{Float64}:
  51.969   74.7366
 112.105  164.074
```

---

```julia
julia> D, P = LinearAlgebra.eigen(A)
LinearAlgebra.Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
values:
2-element Vector{Float64}:
 -0.3722813232690143
  5.372281323269014
vectors:
2×2 Matrix{Float64}:
 -0.824565  -0.415974
  0.565767  -0.909377
```
    
```julia
julia> P * LinearAlgebra.Diagonal(exp.(D)) * inv(P)
2×2 Matrix{Float64}:
  51.969   74.7366
 112.105  164.074
```

#bibliography("refs.bib")