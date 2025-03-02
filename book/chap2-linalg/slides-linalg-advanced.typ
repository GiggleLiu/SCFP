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
  title: [Matrix computation: Advanced topics],
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


== Pivoting

== Flops