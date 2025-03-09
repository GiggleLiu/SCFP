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


= Faster matrix multiplication
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
Q: How many _floating-point operations_ ($+$, $-$, $*$ and $\/$) are needed to compute the product of two $n times n$ matrices?

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

== Strassen's algorithm

Strassen's algorithm@Strassen1969 is a divide-and-conquer algorithm for matrix multiplication in time $O(n^2.81)$.

=== Example
$
A = mat(
  a_(1 1), a_(1 2);
  a_(2 1), a_(2 2)
), quad

B = mat(
  b_(1 1), b_(1 2);
  b_(2 1), b_(2 2)
)
$

The product $C = A times B$ can be computed using the following steps:

== 1. Compute the following seven products:
   $
   &M_1 = (a_(1 1) + a_(2 2))(b_(1 1) + b_(2 2))\
   &M_2 = (a_(2 1) + a_(2 2))b_(1 1)\
   &M_3 = a_(1 1)(b_(1 2) - b_(2 2))\
   &M_4 = a_(2 2)(b_(2 1) - b_(1 1))\
   &M_5 = (a_(1 1) + a_(1 2))b_(2 2)\
   &M_6 = (a_(2 1) - a_(1 1))(b_(1 1) + b_(1 2))\
   &M_7 = (a_(1 2) - a_(2 2))(b_(2 1) + b_(2 2))
   $

== 2. Compute the submatrices of $C$:
   $
   &c_(1 1) = M_1 + M_4 - M_5 + M_7\
   &c_(1 2) = M_3 + M_5\
   &c_(2 1) = M_2 + M_4\
   &c_(2 2) = M_1 - M_2 + M_3 + M_6
   $

Finally, combine the submatrices to form the final product matrix $C$:
   $
   C = mat(
     c_(1 1), c_(1 2);
     c_(2 1), c_(2 2)
   )
   $

== Implementation
#columns(2, text(12pt)[```julia
function strassen(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    n = size(A, 1)
    n == 1 && return A * B
    @assert iseven(n) && size(A) == size(B) "matrix sizes must be even and equal"

    m = div(n, 2)
    A11, A12 = A[1:m, 1:m], A[1:m, m+1:n]
    A21, A22 = A[m+1:n, 1:m], A[m+1:n, m+1:n]
    B11, B12 = B[1:m, 1:m], B[1:m, m+1:n]
    B21, B22 = B[m+1:n, 1:m], B[m+1:n, m+1:n]

    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = similar(A)
    C[1:m, 1:m] = C11
    C[1:m, m+1:n] = C12
    C[m+1:n, 1:m] = C21
    C[m+1:n, m+1:n] = C22

    return C
end

@testset "Strassen's Algorithm" begin
    A = rand(4, 4)
    B = rand(4, 4)
    C = strassen(A, B)
    @test C ≈ A * B
end
```])

==
- _Remark:_ The state of the art algorithm has a time complexity of $O(n^2.37286)$@Alman2024. 
- _Remark:_ The practically favorable one is still the $O(n^3)$ one due to the small overhead.


== Fast Fourier Transform

The Fourier transform is a *linear transformation* widely used in signal processing, image processing and physics.
It transforms a function in the time/space domain into a representation in the _frequency domain_. For a complex-valued function $f(x)$, the Fourier transform and its inverse transform are defined as:

$ g(u) = cal(F)(f(x)) = integral_(-infinity)^infinity e^(-2 pi i u x) f(x) dif x\
f(x) = cal(F)^(-1)(g(u)) = 1/(2pi) integral_(-infinity)^infinity e^(2 pi i u x) g(u) dif u
$
Here, $u$ represents frequency in the _frequency domain_, while $x$ represents position/time in the _physical domain_.

== Discrete Fourier Transform
#figure(canvas({
  import draw: *
  let s(it) = text(12pt)[#it]
  let r = 1.5
  circle((0, 0), radius: r, name: "circle")
  let n = 8
  for i in range(n){
    let (x, y) = (calc.cos(i * 2 * calc.pi/n), calc.sin(i * 2 * calc.pi/n))
    line((x*r, y*r), (1.1*x*r, 1.1*y*r), name: "line" + str(i))
  }
  content((0, r + 0.4), s[$0 = 2 pi$])
  content((0, 0), s[$omega = e^(-2pi i\/n)$])
}),
)
When working with discrete data over a finite domain, we use the discrete Fourier transform (DFT). For a vector $bold(x) = (x_0, x_1, dots, x_(n-1))$ of length $n$, the DFT is defined as:
$ y_k = sum_(j=0)^(n-1) x_j e^(-2pi i k j\/n) = sum_(j=0)^(n-1) x_j omega^(k j) = F_n bold(x) $ 

where $omega = e^(-2pi i\/n)$ is the primitive $n$th root of unity. $F_n$ is the _DFT matrix_.

== DFT matrix
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

== Implementation
The Julia package `FFTW.jl` contains an extremely efficient implementation of FFT.

#box(text(14pt)[```julia
julia> using FFTW
julia> using LinearAlgebra

julia> function dft_matrix(n::Int)  # the definition of the DFT matrix
           ω = exp(-2π*im/n)
           return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
       end

julia> n = 3
julia> Fn = dft_matrix(n)
julia> Fn * (Fn'/3) ≈ I(n)    # output: true

julia> x = ones(n)            # a uniform vector
julia> y = FFTW.fft(copy(x))  # a uniform function is transformed to delta function
julia> y ≈ dft_matrix(n) * x  # output: true
```])

== Application 1: Fast polynomial multiplication

Given two polynomials $p(x)$ and $q(x)$ of degree $n-1$:

$ p(x) = sum_(k=0)^(n-1) a_k x^k, quad q(x) = sum_(k=0)^(n-1) b_k x^k $

Their product is a polynomial of degree $2n-2$:

$ p(x)q(x) = sum_(k=0)^(2n-2) c_k x^k $

The naive approach to compute the coefficients $c_k$ requires $O(n^2)$ operations. However, using the Fast Fourier Transform (FFT), we can compute this product in $O(n log n)$ time by leveraging a fundamental property: multiplication in the frequency domain corresponds to convolution in the time domain.

== Algorithm: Fast Polynomial Multiplication

1. Evaluate $p(x)$ and $q(x)$ at $2n$ roots of unity $omega^0, dots, omega^(2n-1)$ using FFT, where $omega = e^(-2pi i\/2n)$.
2. Multiply the values pointwise:
    $ (p compose q)(omega^j) = p(omega^j) q(omega^j) $ for $j = 0, dots, 2n-1$.
3. Use the inverse FFT to recover the coefficients $c_0, c_1, dots, c_(2n-2)$ of the product polynomial.
- _Remark:_ The FFT and inverse FFT each take $O(n log n)$ time, while the pointwise multiplication takes $O(n)$ time, giving a total complexity of $O(n log n)$.
- _Remark:_ This algorithm generalizes to computing vector convolutions. For vectors $a = (a_0, dots, a_(n-1))$ and $b = (b_0, dots, b_(n-1))$, their convolution $c = (c_0, dots, c_(2n-2))$ is: $c_j = sum_(k=0)^j a_k b_(j-k), quad j = 0,dots,2n-2$


== Implementation

#box(text(16pt)[```julia
using Polynomials
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

= Triangular linear systems

== Triangular linear systems
Only $O(n^2)$ time is needed to solve a system of linear equations with a lower triangular matrix
$ L x = b $
where $L in RR^(n times n)$ is a lower triangular matrix defined as
$
L = mat(
  l_(1 1), 0, dots.h, 0;
  l_(2 1), l_(2 2), dots.h, 0;
  dots.v, dots.v, dots.down, dots.v;
  l_(n 1), l_(n 2), dots.h, l_(n n)
)
$
== Forward-substitution
The forward substitution can be summarized to the following algorithm
$
x_1 = b_1\/l_(1 1), quad x_i = (b_i - sum_(j=1)^(i-1) l_(i j) x_j)\/l_(i i), quad i=2, ..., n
$

*Example: forward substitution*

Consider the following system of lower triangular linear equations:
$
L x = b quad "where" quad L = mat(
  3, 0, 0;
  2, 5, 0;
  1, 4, 2
), quad x = mat(
  x_1;
  x_2;
  x_3
), quad b = mat(
  9;
  12;
  13
)
$

==
Following the forward substitution algorithm, we solve for each component of $x$ sequentially:
1. First equation ($i=1$): $3x_1 = 9 arrow.r x_1 = 9/3 = 3$
2. Second equation ($i=2$): $2x_1 + 5x_2 = 12 arrow.r x_2 = (12 - 2 times 3)/5 = (12 - 6)/5 = 1.2$
3. Third equation ($i=3$): $x_1 + 4x_2 + 2x_3 = 13 arrow.r x_3 = (13 - 1 times 3 - 4 times 1.2)/2 = (13 - 3 - 4.8)/2 = 2.6$

Therefore, the solution to the system is:
$
x = mat(
  3;
  1.2;
  2.6
)
$

You can verify this is correct by multiplying $L x$ to get $b$.

== Algorithm: Forward Substitution

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

== Implementation
#box(text(14pt)[```julia
function forward_substitution!(l::AbstractMatrix, b::AbstractVector)
    n = length(b)
    @assert size(l) == (n, n) "size mismatch"
    x = zero(b)
    # loop over columns
    for j = 1:n
        # stop if matrix is singular
        if iszero(l[j, j])
            error("The lower triangular matrix is singular!")
        end
        # compute solution component
        x[j] = b[j] / l[j, j]
        for i = j+1:n
            # update right hand side
            b[i] = b[i] - l[i, j] * x[j]
        end
    end
    return x
end
```])

= General linear systems
== Gaussian Elimination
Gaussian elimination is a method for finding the $L U$ decomposition of a matrix:
$
A = L U
$
where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix.

*Example: LU decomposition using Gaussian elimination*

Consider the matrix
$
A = mat(
  1, 2, 2;
  4, 4, 2; 
  4, 6, 4
)
$

==
Let's perform LU decomposition step by step using Gaussian elimination:

1. First step ($k=1$): eliminate entries below $a_(1 1)$
   - Compute multipliers: $m_2 = a_(2 1)\/a_(1 1) = 4$, $m_3 = a_(3 1)\/a_(1 1) = 4$
   - Subtract $m_2$ times row 1 from row 2:
   $
   mat(
     1, 2, 2;
     4, 4, 2;
     4, 6, 4
   ) arrow.r mat(
     1, 2, 2;
     0, -4, -6;
     4, 6, 4
   )
   $
   - Subtract $m_3$ times row 1 from row 3:
   $
   mat(
     1, 2, 2;
     0, -4, -6;
     0, -2, -4
   )
   $

2. Second step ($k=2$): eliminate entries below $a_(2 2)$
   - Compute multiplier: $m_3 = (-2)\/(-4) = 1/2$
   - Subtract $m_3$ times row 2 from row 3:
   $
   mat(
     1, 2, 2;
     0, -4, -6;
     0, -2, -4
   ) arrow.r mat(
     1, 2, 2;
     0, -4, -6;
     0, 0, -1
   ) = U
   $

The lower triangular matrix $L$ contains the multipliers used in the elimination:
$
L = mat(
  1, 0, 0;
  4, 1, 0;
  4, 1/2, 1
)
$
You can verify that $L U = A$:
$
mat(
  1, 0, 0;
  4, 1, 0;
  4, 1/2, 1
) mat(
  1, 2, 2;
  0, -4, -6;
  0, 0, -1
) = mat(
  1, 2, 2;
  4, 4, 2;
  4, 6, 4
)
$

== Implementation
#columns(2, text(14pt)[```julia
function lufact!(a::AbstractMatrix)
    n = size(a, 1)
    @assert size(a, 2) == n "size mismatch"
    m = zero(a)
    m[1:n+1:end] .+= 1
    # loop over columns
    for k=1:n-1
        # stop if pivot is zero
        if iszero(a[k, k])
            error("Gaussian elimination fails!")
        end
        # compute multipliers for current column
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
        end
        # apply transformation to remaining sub-matrix
        for j=k+1:n
            for i=k+1:n
                a[i,j] -= m[i,k] * a[k, j]
            end
        end
    end
    return m, triu!(a)
end

lufact(a::AbstractMatrix) = lufact!(copy(a))

@testset "LU factorization" begin
    a = randn(4, 4)
    L, U = lufact(a)
    @test istril(L)
    @test istriu(U)
    @test L * U ≈ a
end
```
])

== Cholesky Decomposition

Given a positive definite symmetric matrix $A in RR^(n times n)$, the Cholesky decomposition is formally defined as
$
A = L L^T,
$
where $L$ is an upper triangular matrix.

The implementation of Cholesky decomposition is similar to LU decomposition.

```julia
function chol!(a::AbstractMatrix)
    n = size(a, 1)
    @assert size(a, 2) == n
    for k=1:n
        a[k, k] = sqrt(a[k, k])
        for i=k+1:n
            a[i, k] = a[i, k] / a[k, k]
        end
        for j=k+1:n
            for i=k+1:n
                a[i,j] = a[i,j] - a[i, k] * a[j, k]
            end
        end
    end
    return a
end
```

```julia
@testset "cholesky" begin
    n = 10
    Q, R = qr(randn(10, 10))
    a = Q * Diagonal(rand(10)) * Q'
    L = chol!(copy(a))
    @test tril(L) * tril(L)' ≈ a
end
```



== Pivoting technique

The above Gaussian elimination process is numerically unstable if $A$ has a diagonal entry that is close to zero.
```julia
            m[i, k] = a[i, k] / a[k, k]
```

LU factoriaztion with row pivoting is defined as
$
P A = L U
$
where $P$ is a permutation matrix.

*Pivot*: At each step, we select the element with the largest absolute value in the current column as the pivot. We then swap the row containing this pivot with the current row before performing elimination.


== Implementation
#columns(2, text(13pt, [```julia
function lufact_pivot!(a::AbstractMatrix)
  n = size(a, 1)
  @assert size(a, 2) == n "size mismatch"
  m = zero(a)
  P = collect(1:n)
  # loop over columns
  @inbounds for k=1:n-1
    # search for pivot in current column
    val, p = findmax(x->abs(a[x, k]), k:n)
    p += k-1
    if p != k # swap rows k and p of matrix A
      for col = 1:n
        a[k, col], a[p, col] = a[p, col], a[k, col]
      end
      # swap rows k and p of matrix M
      for col = 1:k-1
        m[k, col], m[p, col] = m[p, col], m[k, col]
      end
      P[k], P[p] = P[p], P[k]
    end
    iszero(a[k, k]) && continue
    # compute multipliers for current column
    m[k, k] = 1
    for i = k+1:n
      m[i, k] = a[i, k] / a[k, k]
    end
    # apply transformation to remaining sub-matrix
    for j = k+1:n
      akj = a[k, j]
      for i = k+1:n
        a[i, j] -= m[i, k] * akj
      end
    end
  end
  m[n, n] = 1
  return m, triu!(a), P
end
```]))

== Complete pivoting
Complete pivoting extends partial pivoting by allowing both row and column permutations. The LU factorization with complete pivoting is defined as
$
P A Q = L U
$
where $P$ and $Q$ are permutation matrices that reorder both rows and columns.
- _Remark_: While complete pivoting provides superior numerical stability compared to partial pivoting, its implementation is more complex and computationally expensive. In practice, partial pivoting usually provides sufficient numerical stability for most applications while being simpler and faster to compute.

= Avoid Loss of Orthogonality

== (Modified) Gram-Schmidt Orthogonalization
The Gram-Schmidt orthogonalization is the simplest method to compute the QR factorization of a matrix $A$ by iteratively constructing orthonormal columns of $Q$ and the corresponding entries of the upper triangular matrix $R$. 

For each column $k$, we:
1. Take the $k$-th column of $A$, denoted as $a_k$
2. Subtract its projections onto all previous orthonormal vectors $q_1,dots,q_(k-1)$
3. Normalize the result to get $q_k$

This can be expressed mathematically as:

$
q_k = (a_k - sum_(i=1)^(k-1) r_(i k)q_i)\/r_(k k)
$

where $r_(i k) = q_i^T a_k$ represents the projection coefficients that become entries of $R$, and $r_(k k)$ is the normalization factor.
#columns(2, text(14pt, [```julia
function gram_schmidt(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    R[1, 1] = norm(view(A, :, 1))
    Q[:, 1] .= view(A, :, 1) ./ R[1, 1]
    for k = 2:n
        Q[:, k] .= view(A, :, k)
        # project z to span(A[:, 1:k-1])⊥
        for j = 1:k-1
            R[j, k] = view(Q, :, j)' * view(A, :, k)
            Q[:, k] .-= view(Q, :, j) .* R[j, k]
        end
        # normalize the k-th column
        R[k, k] = norm(view(Q, :, k))
        Q[:, k] ./= R[k, k]
    end
    return Q, R
end
```]))

While mathematically equivalent, the modified version is more numerically stable:

For each column $k = 1,dots,n$:
1. Compute the normalization factor $r_(k k) = norm(a_k)$
2. Normalize to obtain $q_k = a_k \/ r_(k k)$
3. For remaining columns $j = k+1,dots,n$:
   - Compute projection coefficient $r_(k j) = q_k^T a_j$ 
   - Update column $a_j = a_j - r_(k j)q_k$ to remove $q_k$ component

== Implementation
#columns(2, text(14pt, [```julia
function modified_gram_schmidt!(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    for k = 1:n
        R[k, k] = norm(view(A, :, k))
        Q[:, k] .= view(A, :, k) ./ R[k, k]
        for j = k+1:n
            R[k, j] = view(Q, :, k)' * view(A, :, j)
            A[:, j] .-= view(Q, :, k) .* R[k, j]
        end
    end
    return Q, R
end
```]))


== Householder Reflection

A key building block for computing QR factorization is the Householder reflection. For any nonzero vector $v in RR^m$, a Householder reflection matrix $P$ takes the form:
$
P = I - beta v v^T, quad beta = (2)/(v^T v)
$
where $I$ is the identity matrix. This matrix has two important properties:
- It is symmetric: $P = P^T$ 
- It is orthogonal: $P^T P = P P^T = I$

== Zero out a vector
Given a vector $x$, we can construct a Householder matrix $H$ that maps $x$ to a multiple of $e_1 = (1, 0, dots, 0)^T$ via:
#figure(canvas({
    import draw: *
    let theta = 2*calc.pi/3
    let nm = 3
    let s(it) = text(12pt)[#it]
    circle((0, 0), radius: 2)
    line((0, 0), (2, 0), mark: (end: "straight"))
    line((0, 0), (nm, 0), mark: (end: "straight"))
    content((1, 0.3), s[$e_1$])
    content((nm, 0.3), s[$H x$])
    line((0, 0), (calc.cos(theta) * nm, nm * calc.sin(theta)), mark: (end: "straight"), name: "x")
    line((0, 0), (calc.cos(theta/2) * nm, nm * calc.sin(theta/2)), stroke: (dash: "dashed"), mark: (end: "straight"), name: "y")
    content((rel : (0.4, -0.1), to: "x.end"), s[$x$])
    content((rel : (0.2, 0.1), to: "y.end"), s[$v$])
    bezier((rel : (-0.4 * calc.sin(theta/2), 0.4 * calc.cos(theta/2)), to: "y.mid"), (rel : (0.4 * calc.sin(theta/2), -0.4 * calc.cos(theta/2)), to: "y.mid"), (rel : (0.4 * calc.cos(theta/2), 0.4 * calc.sin(theta/2)), to: "y.mid"), name: "H", mark: (end: "straight", start: "straight"))
    content((rel : (0.1, -0.2), to: "H.end"), s[Mirror])
}))

$
v &= x plus.minus ||x||_2 e_1, quad H &= I - beta v v^T, quad beta = (2)/(v^T v)
$
where the sign is chosen to avoid cancellation errors. By reflecting $x$ onto $e_1$, we can zero out the elements of $x$ below the first entry.

== Implementation
#columns(2, text(14pt, [```julia
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}
    β::T
end
function HouseholderMatrix(v::Vector{T}) where T
    HouseholderMatrix(v, 2/norm(v, 2)^2)
end

# array interfaces: `size`, `getindex`.
Base.size(A::HouseholderMatrix) = (length(A.v), length(A.v))
Base.size(A::HouseholderMatrix, i::Int) = i == 1 || i == 2 ? length(A.v) : 1
function Base.getindex(A::HouseholderMatrix, i::Int, j::Int)
    (i == j ? 1 : 0) - A.β * A.v[i] * conj(A.v[j])
end

# Householder matrix is unitary
Base.inv(A::HouseholderMatrix) = A
# Householder matrix is Hermitian
Base.adjoint(A::HouseholderMatrix) = A

# Left and right multiplication
function left_mul!(B, A::HouseholderMatrix)
    B .-= (A.β .* A.v) * (A.v' * B)
    return B
end
function right_mul!(A, B::HouseholderMatrix)
    A .= A .- (A * (B.β .* B.v)) * B.v'
    return A
end
```
]))

#box(text(16pt, [```julia
using LinearAlgebra, Test
@testset "householder property" begin
    v = randn(3)
    H = HouseholderMatrix(v)
    # symmetric
    @test H' ≈ H
    # reflexive
    @test H^2 ≈ I
    # orthogonal
    @test H' * H ≈ I
end
```]))

== Householder matrix that projects a vector to $e_1$
Let us define a function to compute the Householder matrix that projects a vector to $e_1$.
#box(text(16pt, [```julia
function householder_e1(v::AbstractVector{T}) where T
    v = copy(v)
    v[1] -= norm(v, 2)
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end
```]))

#box(text(16pt, [```julia
A = Float64[1 2 2; 4 4 2; 4 6 4]
hm = householder_e1(view(A,:,1))
hm * A
```]))

== QR factorization with Householder reflections

The QR factorization using Householder reflections works by successively applying Householder matrices to transform A into an upper triangular matrix R:

$
H_n H_(n-1) dots H_2 H_1 A = R
$

where $R$ is upper triangular. The orthogonal matrix $Q$ is then defined as the product of the Householder reflections:

$
Q = H_1 H_2 dots H_n
$

Since each $H_k$ is both orthogonal and self-adjoint ($H_k = H_k^dagger$), $Q$ is orthogonal and $A = Q R$ gives the desired QR factorization.

== Implementation
- _Remark_: The Householder reflection is numerically stable, we have $||Q^dagger Q - I||_2 approx u$ for $u$ the machine precision. While for the Modified Gram-Schmidt process, it is $u kappa(A)$ for $kappa(A)$ the condition number of $A$.
#columns(2, text(14pt, [```julia
function householder_qr!(Q::AbstractMatrix{T}, a::AbstractMatrix{T}) where T
    m, n = size(a)
    @assert size(Q, 2) == m
    if m == 1
        return Q, a
    else
        # apply householder matrix
        H = householder_e1(view(a, :, 1))
        left_mul!(a, H)
        # update Q matrix
        right_mul!(Q, H')
        # recurse
        householder_qr!(view(Q, 1:m, 2:m), view(a, 2:m, 2:n))
    end
    return Q, a
end
```]))

== Givens Rotations
Given's rotation is another way to perform QR factorization.
The Givens matrix has the form:

$
G_(i j)(theta) = mat(
  1, dots.h, 0, dots.h, 0, dots.h, 0;
  dots.v, dots.down, dots.v, dots.v, dots.v, dots.v, dots.v;
  0, dots.h, c, dots.h, -s, dots.h, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v, dots.v, dots.v;
  0, dots.h, s, dots.h, c, dots.h, 0;
  dots.v, dots.v, dots.v, dots.v, dots.v, dots.down, dots.v;
  0, dots.h, 0, dots.h, 0, dots.h, 1
)
$

When applied to a vector, it performs a rotation in the $(i,j)$ plane:

$
g = mat(
  cos theta, -sin theta;
  sin theta, cos theta
)
$

== Zero out an element

#figure(canvas({
    import draw: circle, line, content, bezier
    let theta = 2*calc.pi/3
    let nm = 3
    let s(it) = text(12pt)[#it]
    circle((0, 0), radius: 2)
    line((0, 0), (2, 0), mark: (end: "straight"))
    line((0, 0), (nm, 0), mark: (end: "straight"))
    content((1, 0.3), s[$e_1$])
    content((nm+0.2, 0.4), s[$g(-\u{2220} x e_1) x$])
    line((0, 0), (calc.cos(theta) * nm, nm * calc.sin(theta)), mark: (end: "straight"), name: "x")
    circle((calc.cos(theta/2) * 0.2, calc.sin(theta/2) * 0.2), radius: 0.0, name: "y")
    content((rel : (0.4, -0.1), to: "x.end"), s[$x$])
    bezier((rel : (-0.4 * calc.sin(theta/2), 0.4 * calc.cos(theta/2)), to: "y"), (rel : (0.4 * calc.sin(theta/2), -0.4 * calc.cos(theta/2)), to: "y"), (rel : (0.4 * calc.cos(theta/2), 0.4 * calc.sin(theta/2)), to: "y"), name: "H", mark: (end: "straight"))
}))


#box(text(16pt, [```julia
rotation_matrix(angle) = [cos(angle) -sin(angle); sin(angle) cos(angle)]
```]))

#box(text(16pt, [```julia
angle = π/4
initial_vector = [1.0, 0.0]
final_vector = rotation_matrix(angle) * initial_vector
# eliminating the y element
atan(0.1, 0.5)
initial_vector = randn(2)
angle = atan(initial_vector[2], initial_vector[1])
final_vector = rotation_matrix(-angle) * initial_vector
# check if the element is zero
@test final_vector[2] ≈ 0 atol=1e-8
```]))

== QR Factorization with Givens Rotations

= Do not Implement Your Own Linear Algebra

While implementing linear algebra algorithms from scratch is educational, in production code you should use established libraries like BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package). These libraries provide:

1. Highly optimized implementations that leverage hardware capabilities
2. Numerically stable algorithms tested across many platforms
3. Standardized interfaces used throughout scientific computing

== Basic Linear Algebra Subprograms (BLAS)

BLAS provides fundamental building blocks for linear algebra operations, organized into three levels based on computational complexity:

- Level 1: Vector-vector operations ($O(n)$ complexity)
- Level 2: Matrix-vector operations ($O(n^2)$ complexity) 
- Level 3: Matrix-matrix operations ($O(n^3)$ complexity)


== BLAS level 1 (time complexity: $O(n)$)
#box([```
axpy!(n, a, X, incx, Y, incy)
```])
Overwrite `Y` with `a * X + Y`. Returns `Y`.

== BLAS level 2 (time complexity: $O(n^2)$)
#box([```
gemv!(tA, alpha, A, x, beta, y)
```])
Update the vector `y` as `alpha * A * x + beta * y` or `alpha * A' * x + beta * y` according to `tA`. `alpha` and `beta` are scalars. Return the updated `y`.

#box([```
trsv!(ul, tA, dA, A, b)
```])
Overwrite `b` with the solution to `A * x = b` or one of the other two variants determined by `tA` and `ul`. `dA` determines if the diagonal values are read or are assumed to be all ones. Return the updated `b`.

== BLAS level 3 (time complexity: $O(n^3)$)

#box([```
gemm!(tA, tB, alpha, A, B, beta, C)
```])
Update `C` as `alpha * A * B + beta * C` or `alpha * A' * B + beta * C` or `alpha * B * A + beta * C` or `alpha * B' * A + beta * C` according to `tA` and `tB`. Returns the updated `C`.

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
[orthogonal], [or], [], [op], [], [],
[unitary], [un], [], [up], [], [],
)

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

== LAPACK: Singular Value Decomposition

#box([```
gesvd!(jobu, jobvt, A) -> (U, S, VT)
```])

Finds the singular value decomposition of $A$, $A = U S V^T$. If `jobu = A`, all the columns of $U$ are computed. If `jobvt = A` all the rows of $V^T$ are computed. If `jobu = N`, no columns of $U$ are computed. If `jobvt = N` no rows of $V^T$ are computed. If `jobu = O`, $A$ is overwritten with the columns of (thin) $U$. If `jobvt = O`, $A$ is overwritten with the rows of (thin) $V^T$. If `jobu = S`, the columns of (thin) $U$ are computed and returned separately. If `jobvt = S` the rows of (thin) $V^T$ are computed and returned separately. `jobu` and `jobvt` can't both be `O`.

Returns $U$, $S$, and $V^T$, where $S$ are the singular values of $A$.

== LAPACK: Generalized Eigenvalue Problem

The generalized eigenvalue problem is to find the eigenvalues and eigenvectors of a matrix pair $(A, B)$ such that
$
A z = lambda B z
$

#box([```
ggev!(jobvl, jobvr, A, B) -> (alpha, beta, vl, vr)
```])

Finds the generalized eigendecomposition of $A$ and $B$. If `jobvl = N`, the left eigenvectors aren't computed. If `jobvr = N`, the right eigenvectors aren't computed. If `jobvl = V` or `jobvr = V`, the corresponding eigenvectors are computed.

== LAPACK: Matrix factorizations
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
#box([```
gels!(trans, A, B) -> (F, B, ssr)
```])
Solves the linear equation A \* X = B, transpose(A) \* X = B, or adjoint(A) \* X = B using a QR or LQ factorization. Modifies the matrix/vector B in place with the solution. A is overwritten with its QR or LQ factorization. trans may be one of N (no modification), T (transpose), or C (conjugate transpose). gels! searches for the minimum norm/least squares solution. A may be under or over determined. The solution is returned in B.

#box([```
gesv!(A, B) -> (B, A, ipiv)
```])
Solves the linear equation A \* X = B where A is a square matrix using the LU factorization of A. A is overwritten with its LU factorization and B is overwritten with the solution X. ipiv contains the pivoting information for the LU factorization of A.

== LAPACK Naming Convention

LAPACK routine names follow the pattern: `XYYZZZ`, where:
- `X`: Data type (S=single real, D=double real, C=single complex, Z=double complex)
- `YY`: Matrix type (see table below)
- `ZZZ`: Operation type

= BLAS and LAPACK Routines
== Level 1 BLAS

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

== Level 2 BLAS

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

== Level 3 BLAS
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

== Linear solvers in LAPACK

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

== Matrix factorizations in LAPACK

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

#bibliography("refs.bib")