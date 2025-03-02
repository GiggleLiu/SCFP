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
  title: [Matrix computation: Applications],
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
A = mat(2, 3, -2; 3, 2, 3; 4, -3, 2)
x = vec(x_1, x_2, x_3)
b = vec(1, 2, 3)
A x = b
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
- _Remark_: the normal equation method is numerically *unstable* due to the problem of ill-conditioned matrix $A^T A$ (detailed later).


= Errors and condition number

== Floating-point numbers and relative errors
#timecounter(2)

_Remark_: You will learn why the normal equation method is numerically unstable. *Numerical stability* is a central topic in numerical linear algebra.

- Short answer: #link("https://en.wikipedia.org/wiki/Floating-point_arithmetic", "floating-point numbers") for storing real numbers has limited precision.

#figure(image("images/float.png", width: 60%), caption: text(16pt)[An example of a layout for 32-bit floating point numbers (IEEE 754 standard)], numbering: none)

#box(text(16pt)[```julia
julia> eps(Float64)  # relative precision: 1.22e-16
julia> eps(1e-50)    # 1.187e-66, Q: what is the relative precision?
julia> eps(1e50)     # 2.077e34,  Q: what is the relative precision?
julia> typemax(Float64)  # Inf
julia> prevfloat(Inf)    # 1.798e308
```])

== Stability of floating-point arithmetic

Task: compute $p - sqrt(p^2 + q)$ for $p = 12345678$ and $q = 1$.

- Method 1:
#box(text(16pt)[```julia
p - sqrt(p^2 + q)
```])
- Method 2:
#box(text(16pt)[```julia
q/(p + sqrt(p^2 + q))
```])


== Condition number
#timecounter(2)

- _Condition number_ of a matrix $A$ is defined as $kappa(A) = ||A|| ||A^(-1)|| >=1$. If the condition number is close to 1, the matrix is _well-conditioned_, otherwise it is _ill-conditioned_.

- _Remark_: there are two popular norms for matrices: the Frobenius norm and the $p$-norms. Here, we use the $p=2$ norm for simplicity.
  - Frobenius norm: $||A||_F = sqrt(sum_(i j) |a_(i j)|^2)$
  - $p$-norm: $||A||_p = max_(x != 0) (||A x||_p)/ (||x||_p)$, for $p = 2$, it is the spectral norm $||A||_2 = sigma_1(A)$, the largest _singular value_ of $A$.

==

Problem: The condition number of $A^dagger A$ is the square of the *condition number* of $A$, which can be very large.

#box(text(16pt)[```julia
julia> cond(A)
34.899220365288556

julia> cond(A' * A)
1217.9555821049864
```])

== The QR Decomposition approach
#timecounter(2)

The QR decomposition of a matrix $A in bb(C)^(m times n)$ is a factorization of the form
$
A = Q R
$
where $Q in bb(C)^(m times m)$ is an orthogonal matrix and $R in bb(C)^(m times n)$ is an upper triangular matrix.

In Julia, we can find the QR decomposition of a matrix using the `qr` function.

#box(text(16pt)[```julia
julia> Q, R = qr(A)
LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
Q factor: 10×10 LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}
R factor:
3×3 Matrix{Float64}:
 -3.16228  -7.11512  -22.5312
  0.0       4.54148   20.4366
  0.0       0.0        5.74456

julia> x = R \ (Matrix(Q)' * y)
3-element Vector{Float64}:
  2.3781818181818197
  2.492424242424242
 -0.22121212121212133
```])

== LU Decomposition
#timecounter(2)

```julia
julia> using LinearAlgebra

julia> lures = lu(A)  # pivot rows by default
LinearAlgebra.LU{Float64, Matrix{Float64}, Vector{Int64}}
L factor:
3×3 Matrix{Float64}:
 1.0   0.0       0.0
 0.5   1.0       0.0
 0.75  0.944444  1.0
U factor:
3×3 Matrix{Float64}:
 4.0  -3.0   2.0
 0.0   4.5  -3.0
 0.0   0.0   4.33333
```

```julia
julia> lures.L * lures.U ≈ lures.P * A
true
```

== Forward and backward substitution

```julia
julia> forward = LowerTriangular(lures.L) \ (lures.P * b)
3-element Vector{Float64}:
  3.0
 -0.5
  0.2222222222222222

julia> x = UpperTriangular(lures.U) \ forward
3-element Vector{Float64}:
  0.6666666666666666
 -0.07692307692307693
  0.05128205128205128
```



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

== LAPACK functions

- 
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

= Basic linear algebra subprograms (BLAS)

== Matrix multiplication
Given two matrices $A in bb(C)^(m times n)$ and $B in bb(C)^(n times p)$, the product $C = A B$ is defined as
$
C_(i j) = sum_(k=1)^n A_(i k) B_(k j).
$

Question (Review): Consider $m = n = p$, what is the time complexity of matrix multiplication in terms of $n$?
#enum(
  [$O(n^2)$],
  [$O(n^3)$],
  [$O(n^2.3)$],
  numbering: "A.",
)

(If we have time, I will show you a faster algorithm for matrix multiplication.)

== Floating-point operations
Q: How many _floating-point operations_ are needed to compute the product of two $n times n$ matrices?

- Remark 1: this number measures the *arithmetic complexity* of the matrix multiplication more precisely than the time complexity.
== Matrix multiplication in Julia

Newbies' API:
#box(text(16pt)[```julia
julia> A = rand(1000, 1000);
julia> B = rand(1000, 1000);
julia> @btime C = A * B;
```])

Old drivers's API:
#box(text(16pt)[```julia
julia> using LinearAlgebra
julia> mul!(C, A, B, α, β)
julia> BLAS.gemm!(tA, tB, alpha, A, B, beta, C)  # C = α * A * B + β * C
```])

- `tA`, `tB`: `N` for non-transpose, `T` for transpose, `C` for conjugate transpose.
- `alpha`, `beta`: scalars applied to `A * B` and `C`, respectively.

== More old drivers's API: BLAS level 1 (time complexity: $O(n)$)
#box(text(16pt)[```julia
copy!(n, X, incx, Y, incy)
```])
Copy `n` elements of array `X` with stride `incx` to array `Y` with stride `incy`. Returns `Y`.

- Remark: _stride_ is the distance between consecutive elements of the array.

#box([```
dot(n, X, incx, Y, incy)
```])
Dot product of two vectors consisting of `n` elements of array `X` with stride `incx` and `n` elements of array `Y` with stride `incy`. There are no dot methods for Complex arrays.

#box([```
nrm2(n, X, incx)
```])
2-norm of a vector consisting of `n` elements of array `X` with stride `incx`.

#box([```
axpy!(n, a, X, incx, Y, incy)
```])
Overwrite `Y` with `a * X + Y`. Returns `Y`.

== More old drivers's API: BLAS level 2 (time complexity: $O(n^2)$)
#box([```
gemv!(tA, alpha, A, x, beta, y)
```])
Update the vector `y` as `alpha * A * x + beta * y` or `alpha * A' * x + beta * y` according to `tA`. `alpha` and `beta` are scalars. Return the updated `y`.

#box([```
trsv!(ul, tA, dA, A, b)
```])
Overwrite `b` with the solution to `A * x = b` or one of the other two variants determined by `tA` and `ul`. `dA` determines if the diagonal values are read or are assumed to be all ones. Return the updated `b`.

#box([```
ger!(alpha, x, y, A)
```])
Rank-1 update of the matrix `A` with vectors `x` and `y` as `alpha * x * y' + A`.

== Solving triangular systems - Explained

We use the forward substitution to solve the lower triangular system $L x = b$.

Consider the following 2-by-2 lower triangular system: 
$
mat(l_11, 0; l_21, l_22) mat(x_1; x_2) = mat(b_1; b_2).
$
If $l_11 l_22 != 0$, then the unknowns can be determined sequentially: 
$
&x_1 = b_1 \/ l_11 ,\
&x_2 = (b_2 - l_21 x_1) \/ l_22.
$
This is the 2-by-2 version of an algorithm known as forward substitution. The general procedure is obtained by solving the ith equation in $L x = b$ for $x_i$:

$
x_i = (b_i - sum_(j=1)^(i-1) l_(i j) x_j) \/ l_(i i)
$

#algorithm({
  import algorithmic: *
  Function("ForwardSubstitution", args: ([$bold(L)$], [$bold(b)$], [$bold(n)$]), {
    Assign([$bold(b)(1)$], [$bold(b)(1) \/ L(1, 1)$ #h(2em)#Ic([update $b(1)$])])
    For(cond: [$i = "2:"n$], {
      Assign([$bold(b)(i)$], [$(bold(b)(i) - L(i, "1:"i-1) dot.c bold(b)("1:"i-1)) \/ L(i, i)$ #h(2em)#Ic([update $b(i)$])])
    })
    Return[$bold(b)$]
  })
})

== More old drivers's API: BLAS level 3 (time complexity: $O(n^3)$)

#box([```
syrk!(uplo, trans, alpha, A, beta, C)
```])
Rank-k update of the symmetric matrix `C` as `alpha * A * A' + beta * C` or `alpha * A' * A + beta * C` according to whether `trans` is ‘N’ or ‘T’. When `uplo` is ‘U’ the upper triangle of `C` is updated (‘L’ for lower triangle).

#box([```
symm!(side, ul, alpha, A, B, beta, C)
```])
Update `C` as `alpha * A * B + beta * C` or `alpha * B * A + beta * C` according to `side`. `A` is assumed to be symmetric. Only the `ul` triangle of `A` is used. Returns the updated `C`.

#box([```
gemm!(tA, tB, alpha, A, B, beta, C)
```])
Update `C` as `alpha * A * B + beta * C` or `alpha * A' * B + beta * C` or `alpha * B * A + beta * C` or `alpha * B' * A + beta * C` according to `tA` and `tB`. Returns the updated `C`.

= Linear algebra packages (LAPACK)

== Problems solved by LAPACK
- Systems of linear equations
- Linear least squares problems
- Matrix factorizations
  - LU
  - Cholesky
  - QR
  - Eigenvalue decomposition
  - Singular value decomposition (SVD)
  - Schur and Generalized Schur
- Estimating condition numbers
- Reordering of the Schur factorizations

== Matrix types in BLAS and LAPACK - By Shape

#let h(it) = table.cell(fill: silver, align: center)[#it]
#let c(it) = table.cell(fill: white, align: center)[#it]
#table(columns: (1fr, 1fr, 1fr, 1fr),
h[#text(red)[ge]neral], h[#text(red)[tr]iangular], h[upper #text(red)[H]ei#text(red)[s]enberg], h[#text(red)[t]rape#text(red)[z]oidal],
c[#canvas({
  import draw: *
  rect((0, 0), (3, 3), fill: red)
})],
c[#canvas({
  import draw: *
  rect((0, 0), (3, 3), fill: none)
  line((3, 0), (0, 3), (3, 3), close: true, fill: red)
})],

c[#canvas({
  import draw: *
  rect((0, 0), (3, 3), fill: none)
  line((3, 0),(2, 0),  (0, 2), (0, 3), (3, 3), close: true, fill: red)
})],

c[#canvas({
  import draw: *
  rect((0, 0), (4, 3), fill: none)
  line((4, 0), (3, 0), (0, 3), (4, 3), close: true, fill: red)
})],


h[#text(red)[di]agonal],h[#text(red)[b]i#text(red)[d]iagonal],h[],h[],
c[#canvas({
  import draw: *
  rect((0, 0), (3, 3), fill: none)
  let n = 20
  let delta = 3 / n
  for i in range(n){
    rect((3 - i * delta, i * delta), (3 - (i+1) * delta, (i+1) * delta), fill: red)
  }
})],

c[#canvas({
  import draw: *
  rect((0, 0), (3, 3), fill: none)
  let n = 20
  let delta = 3 / n
  for i in range(n){
    rect((3 - i * delta, i * delta), (3 - (i+1) * delta, (i+1) * delta), fill: red)
  }
  for i in range(n - 1){
    rect((3 - (i+1) * delta, i * delta), (3 - (i+2) * delta, (i+1) * delta), fill: red)
    //rect((3 - i * delta, (i+1) * delta), (3 - (i+1) * delta, (i+2) * delta), fill: red)
  }
})],
)

Note: More general sparse matrix types will be introduced later.

== Matrix types in BLAS and LAPACK - By Symmetry

For a real matrix $A in RR^(n times n)$
#table(columns: (1fr, 1fr, 1fr),
h[#text(red)[s]ymmetric], h[#text(red)[o]rthogonal],  h[S#text(red)[P]D],
c[$A = A^T$], c[$A^T A = I$], c[$forall_(x!=0) x^T A x > 0$],
)
For a complex matrix $A in CC^(n times n)$
#table(columns: (1fr, 1fr, 1fr),
h[#text(red)[H]ermitian], h[#text(red)[u]nitary], h[H#text(red)[P]D],
c[$A = A^dagger$], c[$A^dagger A = I$], c[$forall_(x != 0) x^dagger A x > 0$],
)

==

Columns are storage types, rows are matrix types.
#table(columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
table.header(h[], h[full], h[#text(red)[b]anded], h[#text(red)[p]acked], h[#text(red)[t]ridiag], h[#text(red)[g]eneralized problem]),
[general], [ge], [gb], [], [gt], [gg],
[symmetric], [sy], [sb], [sp], [st],[],
[Hermitian], [he], [hb], [hp], [ht], [],
[SPD/HPD], [po], [pb], [pp], [pt], [],
[triangular], [tr], [tb], [tp], [], [tg],
[upper Hessenberg], [hs], [], [], [], [hg],
[orthogonal], [or], [op], [], [], [],
[unitary], [un], [up], [], [], [],
)

== An example of generalized problem

The generalized eigenvalue problem is to find the eigenvalues and eigenvectors of a matrix pair $(A, B)$ such that
$
A z = lambda B z
$

== LU factorization of a banded matrix

#box([```
gbtrf!(kl, ku, m, AB) -> (AB, ipiv)
```])
Compute the LU factorization of a banded matrix AB. kl is the first subdiagonal containing a nonzero band, ku is the last superdiagonal containing one, and m is the first dimension of the matrix AB. Returns the LU factorization in-place and ipiv, the vector of pivots used.

#box([```
getrf!(A, ipiv) -> (A, ipiv, info)
```])
Compute the pivoted LU factorization of A, A = LU. ipiv contains the pivoting information and info a code which indicates success (info = 0), a singular value in U (info = i, in which case U[i,i] is singular), or an error code (info < 0).



#box([```
gbtrs!(trans, kl, ku, m, AB, ipiv, B)
```])
Solve the equation AB \* X = B. trans determines the orientation of AB. It may be N (no transpose), T (transpose), or C (conjugate transpose). kl is the first subdiagonal containing a nonzero band, ku is the last superdiagonal containing one, and m is the first dimension of the matrix AB. ipiv is the vector of pivots returned from gbtrf!. Returns the vector or matrix X, overwriting B in-place.

#box([```
gelqf!(A, tau)
```])
Compute the LQ factorization of A, A = LQ. tau contains scalars which parameterize the elementary reflectors of the factorization. tau must have length greater than or equal to the smallest dimension of A.

#box([```
geqlf!(A, tau)
```])
Compute the QL factorization of A, A = QL. tau contains scalars which parameterize the elementary reflectors of the factorization. tau must have length greater than or equal to the smallest dimension of A.

Returns A and tau modified in-place.

#box([```
geqrf!(A, tau)
```])
Compute the QR factorization of A, A = QR. tau contains scalars which parameterize the elementary reflectors of the factorization. tau must have length greater than or equal to the smallest dimension of A.

Returns A and tau modified in-place.

#box([```
gerqf!(A, tau)
```])
Compute the RQ factorization of A, A = RQ. tau contains scalars which parameterize the elementary reflectors of the factorization. tau must have length greater than or equal to the smallest dimension of A.

Returns A and tau modified in-place.

#box([```
gels!(trans, A, B) -> (F, B, ssr)
```])
Solves the linear equation A \* X = B, transpose(A) \* X = B, or adjoint(A) \* X = B using a QR or LQ factorization. Modifies the matrix/vector B in place with the solution. A is overwritten with its QR or LQ factorization. trans may be one of N (no modification), T (transpose), or C (conjugate transpose). gels! searches for the minimum norm/least squares solution. A may be under or over determined. The solution is returned in B.

#box([```
gesv!(A, B) -> (B, A, ipiv)
```])
Solves the linear equation A \* X = B where A is a square matrix using the LU factorization of A. A is overwritten with its LU factorization and B is overwritten with the solution X. ipiv contains the pivoting information for the LU factorization of A.


== Strassen's algorithm

#bibliography("refs.bib")