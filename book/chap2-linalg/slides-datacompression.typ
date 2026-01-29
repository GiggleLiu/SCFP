#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": *
#import "@preview/cetz-plot:0.1.2": *
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

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Matrix computation: Data compression],
    subtitle: [],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#title-slide()
#outline-slide()

= Singular value decomposition

== Singular value decomposition

Let $A in RR^(m times n)$ be a matrix, the *singular value decomposition (SVD)* of $A$ is a factorization of the form
$
A = U S V^dagger
$
where $U$ and $V$ are *unitary matrices* (i.e. $U^dagger U = I$ and $V^dagger V = I$), and $S$ is a diagonal matrix with non-negative real numbers on the diagonal. The columns of $U$ and $V$ are the left and right singular vectors of $A$, respectively. The diagonal elements of $S$ are the *singular values* of $A$.

== Application: Principle component analysis
#figure(canvas(length: 0.9cm, {
    import draw: *
    
    // Random points for scatter plot
    let x = (1.2, 3.4, 5.6, 2.1, 7.8, 4.3, 8.9, 6.2, 1.5, 9.1, 3.7, 5.9, 2.4, 7.3, 4.8, 8.2, 6.5, 1.8, 9.4, 3.1)
    let y = (2.3, 4.5, 1.7, 5.8, 3.2, 6.1, 2.9, 4.7, 1.3, 5.4, 3.6, 2.1, 4.9, 1.5, 6.3, 3.8, 2.6, 5.1, 4.2, 1.9)
    
    plot.plot(
        size: (10, 6),
        x-tick-step: 2,
        y-tick-step: 1,
        x-label: [$X$],
        y-label: [$Y$],
        x-min: 0,
        x-max: 11,
        y-min: 0,
        y-max: 7,
        {
        plot.add(
            x.zip(y.map(y => y * 0.2)).map(xy=>(xy.at(0), xy.at(1) + 0.5 * xy.at(0))),
            style: (stroke: none),
            mark: "o",
            mark-style: (fill: blue),
            mark-size: 0.2,
        )
        plot.add(domain: (0, 10), x=>0.53 * x + 0.6, style: (stroke: black))
        }
    )
}))

- Fitted line: $y = 0.53x + 0.6$
- Data compression point of view: only store the $x$-coordinates and the fitted expression.

== Question

- How many data points are needed to approximately represent the coordinate of a star in galaxy?

== Singular value decomposition for data compression

Step 1: given an $n$ dimensional dataset
$
  cal(D) = { bold(x)_1, bold(x)_2, ..., bold(x)_m } subset RR^n,
$

Step 2: arrange the data points into a matrix
$
  D = mat(bold(x)_1, bold(x)_2, ..., bold(x)_m) in RR^(n times m)
$

Step 3: compute the SVD of $D$
$
  D = U S V^dagger = sum_(i=1)^(min(n, m)) sigma_i u_i v_i^dagger,
$
where $sigma_i$ are the singular values of $D$ and $u_i$ and $v_i$ are the left and right singular vectors of $D$, respectively.

== SVD and data compression
Step 4 (compression): assuming the singular values are sorted in descending order, the first $k$ columns of $U$ are the data points in the new basis
$
  cal(D)_("new") = { U_(:, 1), U_(:, 2), ..., U_(:, k) } subset RR^n
$

Step 5 (reconstruction): the data points in the new basis can be reconstructed by
$
  bold(x)_("new") = U_(:, 1) sigma_1 v_1^dagger + U_(:, 2) sigma_2 v_2^dagger + ... + U_(:, k) sigma_k v_k^dagger,
$

- Question 1: What is the compression ratio?
- Question 2: What is the reconstruction error?

== Relate to PCA

Let $A in RR^(2 times m)$ be a dataset of $m$ 2-dimensional data points, the PCA of $A$ is defined as
$
A approx U_(:, 1) sigma_1 V_(:, 1)^dagger,
$

== Issue of SVD
- Computational cost: $O(m n^2)$ for SVD, $O(n^2)$ for each data reconstruction.
- Imagine we have an image dataset of size $1000 times 1000 times "n"$.

== Basis for sparsity

= Fast Fourier Transform

== Fourier transform
#timecounter(2)

The Fourier transform is a *linear transformation* widely used in signal processing, image processing and physics.
It transforms a function in the time/space domain into a representation in the _frequency domain_. For a complex-valued function $f(x)$, the Fourier transform and its inverse transform are defined as:

$ g(u) = cal(F)(f(x)) = integral_(-infinity)^infinity e^(-2 pi i u x) f(x) dif x\
f(x) = cal(F)^(-1)(g(u)) = 1/(2pi) integral_(-infinity)^infinity e^(2 pi i u x) g(u) dif u
$
Here, $u$ represents frequency in the _frequency domain_, while $x$ represents position/time in the _physical domain_.

== Discrete Fourier Transform
#timecounter(1)
When working with discrete data over a finite domain, we use the discrete Fourier transform (DFT). For a vector $bold(x) = (x_0, x_1, dots, x_(n-1))$ of length $n$, the DFT is defined as:
$ y_k = sum_(j=0)^(n-1) x_j e^(-2pi i k j\/n) = sum_(j=0)^(n-1) x_j omega^(k j) = F_n bold(x) $ 

where $omega = e^(-2pi i\/n)$ is the primitive $n$th root of unity. $F_n$ is the _DFT matrix_.

== DFT matrix
#timecounter(1)
$
F_n = mat(
1 , 1 , 1 , dots , 1;
1 , omega , omega^2 , dots , omega^(n-1);
1 , omega^2 , omega^4 , dots , omega^(2n-2);
dots.v , dots.v , dots.v , dots.down , dots.v;
1 , omega^(n-1) , omega^(2n-2) , dots , omega^((n-1)^2)
).
$

The inverse transformation is given by $F_n^dagger x\/n$. The DFT matrix is unitary up to a scale factor: $F_n F_n^dagger = n I$.

== Cooley-Tukey FFT
#timecounter(2)

$ F_n x = mat(
  I_(n/2), D_(n/2);
  I_(n/2), -D_(n/2)
) mat(
  F_(n/2), 0;
  0, F_(n/2)
) vec(x_("odd"), x_("even")) $

where:
- $F_n$ is the DFT matrix of size n
- $D_n = "diag"(1, omega, omega^2, ..., omega^(n-1))$ is a diagonal matrix
- $omega = e^(-2pi i\/n)$ is the primitive nth root of unity
- $x_("odd")$ and $x_("even")$ contain the odd and even indexed elements of x

This decomposition leads to the recurrence relation $T(n) = 2T(n/2) + O(n)$, which solves to $O(n log n)$ total operations.
Here's an implementation of the Cooley-Tukey FFT algorithm:

== Application 1: Fast polynomial multiplication
#timecounter(2)

Given two polynomials $p(x)$ and $q(x)$ of degree $n-1$:

$ p(x) = sum_(k=0)^(n-1) a_k x^k, quad q(x) = sum_(k=0)^(n-1) b_k x^k $

Their product is a polynomial of degree $2n-2$:

$ p(x)q(x) = sum_(k=0)^(2n-2) c_k x^k $

The naive approach to compute the coefficients $c_k$ requires $O(n^2)$ operations. However, using the Fast Fourier Transform (FFT), we can compute this product in $O(n log n)$ time by leveraging a fundamental property: multiplication in the frequency domain corresponds to convolution in the time domain.

== Algorithm: Fast Polynomial Multiplication
#timecounter(2)

1. Evaluate $p(x)$ and $q(x)$ at $2n$ roots of unity $omega^0, dots, omega^(2n-1)$ using FFT, where $omega = e^(-2pi i\/2n)$.
2. Multiply the values pointwise:
    $ (p compose q)(omega^j) = p(omega^j) q(omega^j) $ for $j = 0, dots, 2n-1$.
3. Use the inverse FFT to recover the coefficients $c_0, c_1, dots, c_(2n-2)$ of the product polynomial.

==
- _Remark:_ The FFT and inverse FFT each take $O(n log n)$ time, while the pointwise multiplication takes $O(n)$ time, giving a total complexity of $O(n log n)$.
- _Remark:_ This algorithm generalizes to computing vector convolutions. For vectors $a = (a_0, dots, a_(n-1))$ and $b = (b_0, dots, b_(n-1))$, their convolution $c = (c_0, dots, c_(2n-2))$ is: $c_j = sum_(k=0)^j a_k b_(j-k), quad j = 0,dots,2n-2$


== Live code: Polynomial multiplication
#timecounter(2)

#box(text(16pt)[```julia
using Polynomials, FFTW
p = Polynomial([1, 3, 2, 5, 6])
q = Polynomial([3, 1, 6, 2, 2])
```])

Step 1: Evaluate polynomials at roots of unity. We need $2n-1$ points for a product of degree $2n-2$.

#box(text(16pt)[```julia
n = 5
pvals = fft(vcat(p.coeffs, zeros(4)))  # Pad with zeros to length 2n-1
qvals = fft(vcat(q.coeffs, zeros(4)))

# This FFT is equivalent to evaluating at roots of unity:
ω = exp(-2π*im/(2n-1))
# pvals ≈ map(k->p(ω^k), 0:(2n-1))
```])

==
Step 2: Multiply values pointwise
#box(text(16pt)[```julia
pqvals = pvals .* qvals
```])

Step 3: Inverse FFT to recover coefficients
#box(text(16pt)[```julia
coeffs = real.(ifft(pqvals))  # Result should be real for real polynomials
```])

= BLAS and LAPACK

== BLAS and LAPACK
#timecounter(1)

While implementing linear algebra algorithms from scratch is educational, in production code you should use established libraries like BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package). These libraries provide:

1. _Highly optimized_ implementations that leverage hardware capabilities
2. _Numerically stable_ algorithms tested across many platforms
3. _Standardized_ interfaces used throughout scientific computing

== Basic Linear Algebra Subprograms (BLAS)
#timecounter(1)

BLAS provides fundamental building blocks for linear algebra operations, organized into three levels based on computational complexity:

- Level 1: Vector-vector operations ($O(n)$ complexity)
- Level 2: Matrix-vector operations ($O(n^2)$ complexity) 
- Level 3: Matrix-matrix operations ($O(n^3)$ complexity)

== Matrix multiplication in Julia
#timecounter(1)

Newbies API:
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



== BLAS level 1 (time complexity: $O(n)$)
#timecounter(1)
#box([```
axpy!(n, a, X, incx, Y, incy)
```])
Overwrite `Y` with `a * X + Y`. Returns `Y`.

#box(text(16pt)[```julia
julia> using BenchmarkTools, LinearAlgebra
julia> x, y = randn(1000), randn(1000);

julia> @btime y + 3.0 * x;
479.375 ns (6 allocations: 15.88 KiB)

julia> @btime BLAS.axpy!(3.0, x, y);
125.322 ns (0 allocations: 0 bytes)
```])


== BLAS level 2 (time complexity: $O(n^2)$)
#timecounter(1)
#box([```
gemv!(tA, alpha, A, x, beta, y)
```])
Update the vector `y` as `alpha * A * x + beta * y` or `alpha * A' * x + beta * y` according to `tA`. `alpha` and `beta` are scalars. Return the updated `y`.

#box([```
trsv!(ul, tA, dA, A, b)
```])
Overwrite `b` with the solution to `A * x = b` or one of the other two variants determined by `tA` and `ul`. `dA` determines if the diagonal values are read or are assumed to be all ones. Return the updated `b`.

== BLAS level 3 (time complexity: $O(n^3)$)
#timecounter(1)

#box([```
gemm!(tA, tB, alpha, A, B, beta, C)
```])
Update `C` as `alpha * A * B + beta * C` or `alpha * A' * B + beta * C` or `alpha * B * A + beta * C` or `alpha * B' * A + beta * C` according to `tA` and `tB`. Returns the updated `C`.

== Matrix types in BLAS and LAPACK - By Shape
#timecounter(1)

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
#timecounter(1)

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
#timecounter(1)

Columns are storage types, rows are matrix types.
#table(columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
table.header(h[], h[full], h[#text(red)[b]anded], h[#text(red)[p]acked], h[#text(red)[t]ridiag], h[#text(red)[g]eneralized problem]),
[general], [ge], [gb], [], [gt], [gg],
[symmetric], [sy], [sb], [sp], [st],[],
[Hermitian], [he], [hb], [hp], [ht], [],
[SPD/HPD], [po], [pb], [pp], [pt], [],
[triangular], [tr], [tb], [tp], [], [tg],
[upper Hessenberg], [hs], [], [], [], [hg],
[orthogonal], [or], [], [op], [], [],
[unitary], [un], [], [up], [], [],
)

== Problems solved by LAPACK
#timecounter(1)
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

== LAPACK: Singular Value Decomposition
#timecounter(1)

#box([```
gesvd!(jobu, jobvt, A) -> (U, S, VT)
```])

Finds the singular value decomposition of $A$, $A = U S V^T$. If `jobu = A`, all the columns of $U$ are computed. If `jobvt = A` all the rows of $V^T$ are computed. If `jobu = N`, no columns of $U$ are computed. If `jobvt = N` no rows of $V^T$ are computed. If `jobu = O`, $A$ is overwritten with the columns of (thin) $U$. If `jobvt = O`, $A$ is overwritten with the rows of (thin) $V^T$. If `jobu = S`, the columns of (thin) $U$ are computed and returned separately. If `jobvt = S` the rows of (thin) $V^T$ are computed and returned separately. `jobu` and `jobvt` can't both be `O`.

Returns $U$, $S$, and $V^T$, where $S$ are the singular values of $A$.

== LAPACK: Generalized Eigenvalue Problem
#timecounter(1)

The generalized eigenvalue problem is to find the eigenvalues and eigenvectors of a matrix pair $(A, B)$ such that
$
A z = lambda B z
$

#box([```
ggev!(jobvl, jobvr, A, B) -> (alpha, beta, vl, vr)
```])

Finds the generalized eigendecomposition of $A$ and $B$. If `jobvl = N`, the left eigenvectors aren't computed. If `jobvr = N`, the right eigenvectors aren't computed. If `jobvl = V` or `jobvr = V`, the corresponding eigenvectors are computed.

== LAPACK: Matrix factorizations
#timecounter(1)
#box([```
getrf!(A, ipiv) -> (A, ipiv, info)
```])
Compute the pivoted LU factorization of $A$, $A = L U$. `ipiv` contains the pivoting information and `info` a code which indicates success (`info = 0`), a singular value in $U$ (`info = i`, in which case $U[i,i]$ is singular), or an error code (`info < 0`).

#box([```
geqrf!(A, tau)
```])
Compute the QR factorization of $A$, $A = Q R$. $tau$ contains scalars which parameterize the elementary reflectors of the factorization. $tau$ must have length greater than or equal to the smallest dimension of $A$.

Returns $A$ and $tau$ modified in-place.

== LAPACK: Linear equations
#timecounter(1)
#box([```
gels!(trans, A, B) -> (F, B, ssr)
```])
Solves the linear equation A \* X = B, transpose(A) \* X = B, or adjoint(A) \* X = B using a QR or LQ factorization. Modifies the matrix/vector B in place with the solution. A is overwritten with its QR or LQ factorization. trans may be one of N (no modification), T (transpose), or C (conjugate transpose). gels! searches for the minimum norm/least squares solution. A may be under or over determined. The solution is returned in B.

#box([```
gesv!(A, B) -> (B, A, ipiv)
```])
Solves the linear equation A \* X = B where A is a square matrix using the LU factorization of A. A is overwritten with its LU factorization and B is overwritten with the solution X. ipiv contains the pivoting information for the LU factorization of A.

== LAPACK Naming Convention
#timecounter(1)

LAPACK routine names follow the pattern: `XYYZZZ`, where:
- `X`: Data type (S=single real, D=double real, C=single complex, Z=double complex) (_Remark_: no need in Julia)
- `YY`: Matrix type (e.g. `ge`)
- `ZZZ`: Operation type (e.g. `svd`)

== Quick Reference: Level 1 BLAS
#timecounter(1)

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    h[*Routine*], h[*Operation*],
    [`asum`], [$sum_i |x_i|$],
    [`axpy!`], [$y <- alpha x + y$],
    [`blascopy!`], [$y <- x$],
    [`dot`], [$x^T y$],
    [`dotc`], [$x^dagger y$], 
    [`nrm2`], [$sqrt(sum_i |x_i|^2)$],
    [`rot!`], [$vec(x_i, y_i) <- mat(c, s; -s^*, c) vec(x_i, y_i)$],
    [`scal!`], [$x <- alpha x$],
  ),
)

== Quick Reference: Level 2 BLAS

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    h[*Routine*], h[*Operation*],
    [`gbmv!`], [$y <- alpha A x + beta y$ (banded)],
    [`gemv!`], [$y <- alpha A x + beta y$],
    [`ger!`], [$A <- alpha x y^dagger + A$],
    [`sbmv!`], [$y <- alpha A x + beta y$ (symmetric banded)],
    [`spmv!`], [$y <- alpha A x + beta y$ (symmetric packed)],
    [`symv!`], [$y <- alpha A x + beta y$ (symmetric)],
    [`trmv!`], [$x <- A x$ (triangular)],
    [`trsv!`], [$x <- A^(-1) x$ (triangular)],
  ),
)

== Quick Reference: Level 3 BLAS
#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    h[*Routine*], h[*Operation*],
    [`gemm!`], [$C <- alpha A B + beta C$],
    [`symm!`], [$C <- alpha A B + beta C$ (A symmetric)],
    [`hemm!`], [$C <- alpha A B + beta C$ (A Hermitian)],
    [`syrk!`], [$C <- alpha A A^T + beta C$ (symmetric rank-k update)],
    [`herk!`], [$C <- alpha A A^H + beta C$ (Hermitian rank-k update)],
    [`syr2k!`], [$C <- alpha (A B^T + B A^T) + beta C$ (symmetric rank-2k update)],
    [`her2k!`], [$C <- alpha (A B^H + B A^H) + beta C$ (Hermitian rank-2k update)],
    [`trmm!`], [$B <- alpha A B$ (A triangular)],
    [`trsm!`], [$B <- alpha A^(-1) B$ (A triangular)],
  ),
)

== Quick Reference: Linear solvers in LAPACK

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    h[*Function*], h[*Description*],
    [`gesv!`], [Solve $A X = B$ using LU factorization with partial pivoting],
    [`gtsv!`], [Solve tridiagonal $A X = B$ using LU factorization with partial pivoting],
    [`posv!`], [Solve symmetric/Hermitian positive definite $A X = B$ using Cholesky factorization],
    [`ptsv!`], [Solve symmetric/Hermitian positive definite tridiagonal $A X = B$],
    [`sysv!`], [Solve symmetric $A X = B$ using Bunch-Kaufman factorization],
    [`hesv!`], [Solve Hermitian $A X = B$ using Bunch-Kaufman factorization],
    [`gels!`], [Solve overdetermined/underdetermined $A X = B$ using QR or LQ factorization],
    [`gelsy!`], [Solve overdetermined/underdetermined $A X = B$ using QR factorization with complete pivoting],
    [`gelsd!`], [Solve overdetermined/underdetermined $A X = B$ using SVD with divide-and-conquer],
  ),
)

== Quick Reference: Matrix factorizations in LAPACK

#figure(
  text(16pt, table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    h[*Function*], h[*Description*],
    [`getrf!`], [LU factorization with partial pivoting],
    [`gbtrf!`], [LU factorization of banded matrix with partial pivoting],
    [`gttrf!`], [LU factorization of tridiagonal matrix],
    [`potrf!`], [Cholesky factorization of symmetric/Hermitian positive definite matrix],
    [`pttrf!`], [Factorization of symmetric/Hermitian positive definite tridiagonal matrix],
    [`sytrf!`], [Bunch-Kaufman factorization of symmetric matrix],
    [`hetrf!`], [Bunch-Kaufman factorization of Hermitian matrix],
    [`geqrf!`], [QR factorization],
    [`gelqf!`], [LQ factorization],
    [`gerqf!`], [RQ factorization],
    [`gesvd!`], [Singular Value Decomposition (SVD)],
    [`gesdd!`], [SVD using divide-and-conquer],
    [`syev!`], [Eigenvalue decomposition of symmetric matrix],
    [`geev!`], [Eigenvalue decomposition of general matrix],
  ),
))

= Hands-on

== Introduction to image processing

- Video: https://www.youtube.com/watch?app=desktop&v=DGojI9xcCfg
- Notebooks: https://github.com/mitmath/computational-thinking

== Tasks
1. Run the code demo: https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/ImageProcessing
2. Compress the image in the HSV channel.

==
#bibliography("refs.bib")