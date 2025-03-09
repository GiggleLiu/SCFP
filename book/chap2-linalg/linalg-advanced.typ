#import "../book.typ": book-page
#import "@preview/cetz:0.2.2": *

#show: book-page.with(title: "Matrix Computation (Advanced Topics)")
#let exampleblock(it) = block(fill: rgb("#ffffff"), inset: 1em, radius: 4pt, stroke: black, it)
#set math.equation(numbering: "(1)")

= Matrix Computation (Advanced Topics)

= Sensitivity analysis

== Floating-point numbers and relative errors

In matrix computation, we often encounter the problem of _sensitivity analysis_, which is the study of how the solution of a problem changes when the input data is perturbed. It is crucial issue in scientific computing due to the _round-off errors_ in floating-point arithmetic.
For an numerical unstable algorithm, a small perturbation in the input can lead to a large error in the output.
The layout of 64-bit floating point numbers (IEEE 754 standard) is shown in the figure below.

#figure(canvas(length: 0.8cm, {
  import draw: *
  let (dx, dy) = (1.0, 1.0)
  let s(it) = text(12pt)[#it]
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
We use the notation $"fl"(x) = x (1 + delta)$ to represent the floating-point number $x$ with a relative error $delta$.
$delta$ is upper bounded by the unit roundoff $u$
$
u = "nextfloat"(1.0) - 1.0,  quad abs(delta) <= u
$


```julia
julia> eps(Float64)  # 2.22e-16
julia> eps(1e-50)    # absolute precision: 1.187e-66
julia> eps(1e50)     # absolute precision: 2.077e34
julia> typemax(Float64)  # Inf
julia> prevfloat(Inf)    # 1.798e308
```

We can see that the relative precision $u$ remains approximately the same for different numbers.

#exampleblock[
*Example:  Stability of floating-point arithmetic*

In the following, we compute $p - sqrt(p^2 + q)$ for $p = 12345678$ and $q = 1$ using two different methods.

- Method 1:
```julia
p - sqrt(p^2 + q)       # -4.0978193283081055e-8
```
- Method 2:
```julia
-q/(p + sqrt(p^2 + q))  # -4.0500003321000205e-8
```
The first method is less accurate, because in the first method, we are subtracting two very close large numbers, which can cause a large relative error in the output if the output is close to zero:
$
  epsilon_("rel") = abs("fl"(p) - "fl"(sqrt(p^2 + q)))/(p - sqrt(p^2 + q)) ~ (u p^2)/q.
$
]

== Condition number <sec:condition-number>

Condition number is a measure of the sensitivity of the solution of a problem to the input data. For a linear system $A x = b$, the condition number $kappa(A)$ is defined as $kappa(A) = ||A|| ||A^(-1)|| >=1$. If the condition number is close to 1, the matrix is _well-conditioned_, otherwise it is _ill-conditioned_. Here, we use the $p=2$ norm for simplicity.

#exampleblock[
*Tutorial point: norms of matrices*

There are two popular norms for matrices: the Frobenius norm and the $p$-norms.

- Frobenius norm: $||A||_F = sqrt(sum_(i j) |a_(i j)|^2)$
- $p$-norm: $||A||_p = max_(x != 0) (||A x||_p)/ (||x||_p)$, for $p = 2$, it is the spectral norm $||A||_2 = sigma_1(A)$, the largest _singular value_ of $A$. To see why, consider $A = U S V^dagger$, we have
   $
  (||A x||_2) / (||x||_2) = (||S V^dagger x||_2) / (||x||_2) = (||S y||_2) / (||y||_2) <= lambda_1,
$
  where $y = V^dagger x$. Here, we used the fact that $||U x||_2 = ||x||_2$ for any unitary matrix $U$.
]

When solving the linear system $A x = b$ with a small perturbation $b + delta b$, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

$
  A(x + delta x) = b + delta b\
  arrow.double.r (||delta x||) / (||x||) = (||A^(-1) delta b||) / (||A^(-1) b||) <= (lambda_1(A^(-1))) / (lambda_n (A^(-1))) (||delta b||) / (||b||)
$
where $lambda_1(A^(-1))$ and $lambda_n (A^(-1))$ ($= lambda_1(A)^(-1)$) are the largest and smallest _singular values_ of $A^(-1)$, respectively.
Hence, the relative error of the solution $x$ is at most $kappa(A) times$ the relative error of $b$.

In the previous section, we claim the normal equation is unstable. Revisit the normal equation:
$
x = (A^T A)^(-1) A^T b
$
We effectively solve the linear system: $(A^T A) x = A^T b$. It is numerically less stable since the condition number of $A^dagger A$ is the square of the condition number of $A$. This is no coincidence. Consider $A = U S V^dagger$, we have
$
  A^dagger A = V S^dagger U^dagger U S V^dagger = V S^2 V^dagger.
$
Since $A^dagger A$ is symmetric positive semidefinite, its singular values are the same as its eigenvalues, which are the squared singular values of $A$. Hence, $kappa(A^dagger A) = kappa(A)^2$.


In Julia, we can compute the condition number of a matrix using the `cond` function.

```julia
julia> cond(A)
34.899220365288556

julia> cond(A' * A)
1217.9555821049864
```

= Triangular linear systems

Triangular linear systems are a type of linear system where the coefficient matrix is either upper or lower triangular.
They can be solved efficiently using forward or backward substitution in time $O(n^2)$, where $n$ is the size of the matrix.

== Forward-substitution
Forward substitution is an algorithm used to solve a system of linear equations with a lower triangular matrix
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
The forward substitution can be summarized to the following algorithm
$
x_1 = b_1\/l_(1 1), quad x_i = (b_i - sum_(j=1)^(i-1) l_(i j) x_j)\/l_(i i), quad i=2, ..., n
$

#exampleblock[
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
]

We implement the forward substitution algorithm in Julia language.
```julia
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
```

We can write a test for this algorithm.


```julia
using Test, LinearAlgebra

@testset "back substitution" begin
    # create a random lower triangular matrix
    l = LinearAlgebra.tril(randn(4, 4))
    # target vector
    b = randn(4)
    # solve the linear equation with our algorithm
    x = back_substitution!(l, copy(b))
    @test l * x ≈ b

    # The Julia's standard library `LinearAlgebra` contains a native implementation.
    x_native = LowerTriangular(l) \ b
    @test l * x_native ≈ b
end
```


== Back-substitution

Back substitution is an algorithm used to solve a system of linear equations with an upper triangular matrix
$
U x = b
$
where $U in RR^(n times n)$ is an upper triangular matrix defined as
$
U = mat(
  u_(1 1), u_(1 2), dots.h, u_(1 n);
  0, u_(2 2), dots.h, u_(2 n);
  dots.v, dots.v, dots.down, dots.v;
  0, 0, dots.h, u_(n n)
)
$

The back substitution can be summarized to the following algorithm
$
x_n = b_n\/u_(n n),quad x_i = (b_i - sum_(j=i+1)^(n) u_(i j) x_j)\/u_(i i),quad i=n-1, ..., 1
$

= LU Factorization
== Gaussian Elimination
LU decomposition is a method for solving linear equations that involves breaking down a matrix into lower and upper triangular matrices. The $L U$ decomposition of a matrix $A$ is represented as
$
A = L U,
$
where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix.
Gaussian elimination is a method for finding the $L U$ decomposition of a matrix.

#exampleblock[
*Example: LU decomposition using Gaussian elimination*

Consider the matrix
$
A = mat(
  1, 2, 2;
  4, 4, 2; 
  4, 6, 4
)
$

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
]

In the following, we implement the Gaussian elimination in Julia language.
```julia
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

We can test the performance of our implementation.

```julia
A4 = randn(4, 4)
lufact(A4)
```

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

The above Gaussian elimination process is numerically unstable if $A$ has a diagonal entry that is close to zero. The small diagonal element will be a divisor in the elimination process, which will introduce extremely large numbers in the computed solution.
As we discussed in the previous section, large numbers have a large absolute error that may propagate through the computation.

To avoid this issue, we can use pivoting technique.
== Partial pivoting
LU factoriaztion with row pivoting is defined as
$
P A = L U
$
where $P$ is a permutation matrix.
Pivoting is a crucial technique in Gaussian elimination that helps maintain numerical stability. At each step, we select the element with the largest absolute value in the current column as the pivot. We then swap the row containing this pivot with the current row before performing elimination. This strategy serves two key purposes:

1. It avoids division by very small numbers that could lead to large roundoff errors
2. It reduces error growth during the elimination process

By choosing the largest possible pivot, we minimize the size of the multipliers used in elimination, which helps control error accumulation. Without pivoting, even a well-conditioned system could produce inaccurate results if small pivots are encountered during elimination.

A Julia implementation of the Gaussian elimination with partial pivoting is

```julia
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
        # find index p such that |a_{pk}| ≥ |a_{ik}| for k ≤ i ≤ n
        if p != k
            # swap rows k and p of matrix A
            for col = 1:n
                a[k, col], a[p, col] = a[p, col], a[k, col]
            end
            # swap rows k and p of matrix M
            for col = 1:k-1
                m[k, col], m[p, col] = m[p, col], m[k, col]
            end
            P[k], P[p] = P[p], P[k]
        end
        if iszero(a[k, k])
            # skip current column if it's already zero
            continue
        end
        # compute multipliers for current column
        m[k, k] = 1
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
        end
        # apply transformation to remaining sub-matrix
        for j=k+1:n
            akj = a[k, j]
            for i=k+1:n
                a[i,j] -= m[i,k] * akj
            end
        end
    end
    m[n, n] = 1
    return m, triu!(a), P
end

@testset "lufact with pivot" begin
    n = 5
    A = randn(n, n)
    L, U, P = lufact_pivot!(copy(A))
    pmat = zeros(Int, n, n)
    setindex!.(Ref(pmat), 1, 1:n, P)
    @test L ≈ lu(A).L
    @test U ≈ lu(A).U
    @test pmat * A ≈ L * U
end
```

== Complete pivoting
Complete pivoting extends partial pivoting by allowing both row and column permutations. The LU factorization with complete pivoting is defined as
$
P A Q = L U
$
where $P$ and $Q$ are permutation matrices that reorder both rows and columns. At each step, the pivot is chosen as the largest element in absolute value from the entire remaining submatrix, not just a single column.

While complete pivoting provides superior numerical stability compared to partial pivoting, its implementation is more complex and computationally expensive. In practice, partial pivoting usually provides sufficient numerical stability for most applications while being simpler and faster to compute.

= QR Factorization

The QR factorization is a fundamental matrix decomposition that expresses a matrix $A in RR^(m times n)$ as a product
$
A = Q R
$
where $Q in RR^(m times m)$ is an orthogonal matrix (meaning $Q^T Q = Q Q^T = I$) and $R in RR^(m times n)$ is an upper triangular matrix. This factorization has important applications in solving linear systems, least squares problems, and eigenvalue computations.

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
```julia
function classical_gram_schmidt(A::AbstractMatrix{T}) where T
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

@testset "classical GS" begin
    n = 10
    A = randn(n, n)
    Q, R = classical_gram_schmidt(A)
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end
```

However, the classical Gram-Schmidt process suffers from the loss of orthogonality due to the accumulation of roundoff errors. The modified Gram-Schmidt orthogonalization is a numerically more stable variant of the classical Gram-Schmidt process. While mathematically equivalent, the modified version performs the orthogonalization in a different order that helps reduce the accumulation of roundoff errors.
The algorithm proceeds as follows:

For each column $k = 1,dots,n$:
1. Compute the normalization factor $r_(k k) = norm(a_k)$
2. Normalize to obtain $q_k = a_k \/ r_(k k)$
3. For remaining columns $j = k+1,dots,n$:
   - Compute projection coefficient $r_(k j) = q_k^T a_j$ 
   - Update column $a_j = a_j - r_(k j)q_k$ to remove $q_k$ component

The modified Gram-Schmidt process can be implemented as follows:
```julia
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

@testset "modified GS" begin
    n = 10
    A = randn(n, n)
    Q, R = modified_gram_schmidt!(copy(A))
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end
```


== Householder Reflection

A key building block for computing QR factorization is the Householder reflection. For any nonzero vector $v in RR^m$, a Householder reflection matrix $P$ takes the form:
$
P = I - beta v v^T, quad beta = (2)/(v^T v)
$
where $I$ is the identity matrix. This matrix has two important properties:
- It is symmetric: $P = P^T$ 
- It is orthogonal: $P^T P = P P^T = I$

#figure(canvas({
    import draw: *
    let theta = 2*calc.pi/3
    let nm = 3
    circle((0, 0), radius: 2)
    line((0, 0), (2, 0), mark: (end: "straight"))
    line((0, 0), (nm, 0), mark: (end: "straight"))
    content((1, 0.3), [$e_1$])
    content((nm, 0.3), [$H x$])
    line((0, 0), (calc.cos(theta) * nm, nm * calc.sin(theta)), mark: (end: "straight"), name: "x")
    line((0, 0), (calc.cos(theta/2) * nm, nm * calc.sin(theta/2)), stroke: (dash: "dashed"), mark: (end: "straight"), name: "y")
    content((rel : (0.4, -0.1), to: "x.end"), [$x$])
    content((rel : (0.2, 0.1), to: "y.end"), [$v$])
    bezier((rel : (-0.4 * calc.sin(theta/2), 0.4 * calc.cos(theta/2)), to: "y.mid"), (rel : (0.4 * calc.sin(theta/2), -0.4 * calc.cos(theta/2)), to: "y.mid"), (rel : (0.4 * calc.cos(theta/2), 0.4 * calc.sin(theta/2)), to: "y.mid"), name: "H", mark: (end: "straight", start: "straight"))
    content((rel : (0.1, -0.2), to: "H.end"), [Mirror])
}))

A particularly useful application is using a Householder reflection to zero out elements of a vector. Given a vector $x$, we can construct a Householder matrix $H$ that maps $x$ to a multiple of $e_1 = (1, 0, dots, 0)^T$ via:
$
v &= x plus.minus ||x||_2 e_1 \
H &= I - beta v v^T, quad beta = (2)/(v^T v)
$
where the sign is chosen to avoid cancellation errors. By reflecting $x$ onto $e_1$, we can zero out the elements of $x$ below the first entry.

Let's implement the Householder reflection in Julia:
```julia
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}
    β::T
end
function HouseholderMatrix(v::Vector{T}) where T
    HouseholderMatrix(v, 2/norm(v, 2)^2)
end

# array interfaces
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
The `HouseholderMatrix` type defined above is a custom implementation of a Householder reflection matrix. It inherits from `AbstractArray` to leverage Julia's array interface. The type contains two fields:

- `v`: The reflection vector $v$ that defines the hyperplane of reflection
- `β`: The scaling factor $beta = 2\/(v^T v)$ used in the reflection formula

To make this type work seamlessly with Julia's array operations, we implement the required array interface methods `size` and `getindex`. These methods allow the type to behave like a standard matrix while maintaining an efficient implicit representation. For more details on implementing array interfaces in Julia, see the #link("https://docs.julialang.org/en/v1/manual/interfaces/", "interfaces documentation").
```julia
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
```

Let us define a function to compute the Householder matrix that projects a vector to $e_1$.
```julia
function householder_e1(v::AbstractVector{T}) where T
    v = copy(v)
    v[1] -= norm(v, 2)
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end
```
```julia
A = Float64[1 2 2; 4 4 2; 4 6 4]
hm = householder_e1(view(A,:,1))
hm * A
```

== QR factorization with Householder reflections

The QR factorization using Householder reflections works by successively applying Householder matrices to transform A into an upper triangular matrix R. Let $H_k$ be a Householder reflection that zeros out all elements below the diagonal in column k. The complete transformation can be written as:

$
H_n H_(n-1) dots H_2 H_1 A = R
$

where $R$ is upper triangular. The orthogonal matrix $Q$ is then defined as the product of the Householder reflections:

$
Q = H_1 H_2 dots H_n
$

Since each $H_k$ is both orthogonal and self-adjoint ($H_k = H_k^dagger$), $Q$ is orthogonal and $A = Q R$ gives the desired QR factorization.
```julia
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
```

```julia
@testset "householder QR" begin
    A = randn(3, 3)
    Q = Matrix{Float64}(I, 3, 3)
    R = copy(A)
    householder_qr!(Q, R)
    @info R
    @test Q * R ≈ A
    @test Q' * Q ≈ I
end

A = randn(3, 3)
g = givens_matrix(A, 2, 3)
left_mul!(copy(A), g)
```

== Givens Rotations
Given's rotation is another way to perform QR factorization.
A Givens rotation is a rotation in a plane spanned by two coordinate axes. It can be used to selectively zero out individual elements in a matrix. For a 2D rotation in the $(i,j)$ plane by angle $theta$, the Givens matrix has the form:

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

where $c = cos(theta)$ and $s = sin(theta)$ appear at the intersection of rows and columns $i$ and $j$. When applied to a vector, it performs a rotation in the $(i,j)$ plane:

$
g = mat(
  cos theta, -sin theta;
  sin theta, cos theta
)
$

Let's implement the Givens rotation in Julia:
```julia
rotation_matrix(angle) = [cos(angle) -sin(angle); sin(angle) cos(angle)]
```

```julia
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
```

== QR Factorization with Givens Rotations

QR factorization can be performed using a sequence of Givens rotations. The idea is to systematically eliminate elements below the diagonal, one at a time, working from left to right and bottom to top. For each element we want to zero out, we:

1. Compute the appropriate Givens rotation that will zero out that element
2. Apply the rotation to the matrix
3. Keep track of the product of all rotations to form Q

Here's how we can implement QR factorization using Givens rotations:

```julia
# the Givens rotation in the (i, j) plane
struct GivensMatrix{T} <: AbstractArray{T, 2}
    c::T   # cos(theta)
    s::T   # sin(theta)
    i::Int # row index
    j::Int # column index
    n::Int # size of the matrix
end

Base.size(g::GivensMatrix) = (g.n, g.n)
Base.size(g::GivensMatrix, i::Int) = i == 1 || i == 2 ? g.n : 1
function Base.getindex(g::GivensMatrix{T}, i::Int, j::Int) where T
    @boundscheck i <= g.n && j <= g.n
    if i == j
        return i == g.i || i == g.j ? g.c : one(T)
    elseif i == g.i && j == g.j
        return g.s
    elseif i == g.j && j == g.i
        return -g.s
    else
        return i == j ? one(T) : zero(T)
    end
end

# left multiplication by a Givens matrix
function left_mul!(A::AbstractMatrix, givens::GivensMatrix)
    for col in 1:size(A, 2)
        vi, vj = A[givens.i, col], A[givens.j, col]
        A[givens.i, col] = vi * givens.c + vj * givens.s
        A[givens.j, col] = -vi * givens.s + vj * givens.c
    end
    return A
end

# right multiplication by a Givens matrix
function right_mul!(A::AbstractMatrix, givens::GivensMatrix)
    for row in 1:size(A, 1)
        vi, vj = A[row, givens.i], A[row, givens.j]
        A[row, givens.i] = vi * givens.c + vj * givens.s
        A[row, givens.j] = -vi * givens.s + vj * givens.c
    end
    return A
end

# the Givens matrix that rotates the j-th elements of A to zero
function givens_matrix(A, i, j)
    x, y = A[i, 1], A[j, 1]
    norm = sqrt(x^2 + y^2)
    c = x/norm
    s = y/norm
    return GivensMatrix(c, s, i, j, size(A, 1))
end

# QR factorization using Givens rotations
function givens_qr!(Q::AbstractMatrix, A::AbstractMatrix)
    m, n = size(A)
    if m == 1
        return Q, A
    else
        for k = m:-1:2
            g = givens_matrix(A, k-1, k)
            left_mul!(A, g)
            right_mul!(Q, g)
        end
        givens_qr!(view(Q, :, 2:m), view(A, 2:m, 2:n))
        return Q, A
    end
end
```

```julia
@testset "givens QR" begin
    n = 3
    A = randn(n, n)
    R = copy(A)
    Q, R = givens_qr!(Matrix{Float64}(I, n, n), R)
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end
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

$T(n) = 2 T(n/2) + O(n)$.

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

= Basic Linear Algebra Subprograms (BLAS)

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


== Level 1 BLAS
Level 1 BLAS operations involve vector-vector operations. Here are some common Level 1 BLAS routines:

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
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
  caption: [Common Level 1 BLAS routines. The exclamation mark (!) indicates the function modifies its arguments in-place.]
)

== Level 2 BLAS
Level 2 BLAS operations involve matrix-vector operations. Here are some common Level 2 BLAS routines:

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
  caption: [Common Level 2 BLAS routines. The exclamation mark (!) indicates the function modifies its arguments in-place.]
)

== Level 3 BLAS
Level 3 BLAS operations involve matrix-matrix operations. Here are some common Level 3 BLAS routines:
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
  caption: [Common Level 3 BLAS routines. The exclamation mark (!) indicates the function modifies its arguments in-place.]
)

= LAPACK

== Linear solvers

The following table lists common LAPACK routines for solving linear equations.

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
  caption: [Common LAPACK linear solvers. The exclamation mark (!) indicates the function modifies its arguments in-place.]
)

The following table lists common LAPACK routines for matrix factorizations.

#figure(
  table(
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
  caption: [Common LAPACK matrix factorization routines. The exclamation mark (!) indicates the function modifies its arguments in-place.]
)
== Naming scheme

#figure(
  table(
    columns: (auto, auto),
    inset: 5pt,
    align: left,
    h[*Matrix type*], h[*Description*],
    [BD], [bidiagonal],
    [DI], [diagonal],
    [GB], [general band],
    [GE], [general (i.e., unsymmetric, in some cases rectangular)],
    [GG], [general matrices, generalized problem (i.e., a pair of general matrices)],
    [GT], [general tridiagonal],
    [HB], [complex Hermitian band],
    [HE], [complex Hermitian],
    [HG], [upper Hessenberg matrix, generalized problem (i.e a Hessenberg and a triangular matrix)],
    [HP], [complex Hermitian, packed storage],
    [HS], [upper Hessenberg],
    [OP], [real orthogonal, packed storage],
    [OR], [real orthogonal],
    [PB], [symmetric or Hermitian positive definite band],
    [PO], [symmetric or Hermitian positive definite],
    [PP], [symmetric or Hermitian positive definite, packed storage],
    [PT], [symmetric or Hermitian positive definite tridiagonal],
    [SB], [real symmetric band],
    [SP], [symmetric, packed storage],
    [ST], [real symmetric tridiagonal],
    [SY], [symmetric],
    [TB], [triangular band],
    [TG], [triangular matrices, generalized problem (i.e., a pair of triangular matrices)],
    [TP], [triangular, packed storage],
    [TR], [triangular (or in some cases quasi-triangular)],
    [TZ], [trapezoidal],
    [UN], [complex unitary],
    [UP], [complex unitary, packed storage]
  ),
  caption: [Matrix types in the LAPACK naming scheme]
)

#bibliography("refs.bib")

= Appendix
== The elementary elimination matrix

An elementary elimination matrix is a matrix that is used in the process of Gaussian elimination to transform a system of linear equations into an equivalent system that is easier to solve. It is a square matrix that is obtained by performing a single elementary row operation on the identity matrix.

$
(M_k)_(i j) = cases(
  delta_(i j) & "if" i = j,
  - a_(i k)\/a_(k k) & "if" i > j "and" j = k,
  0 & "otherwise"
)
$

Let $A = (a_(i j))$ be a square matrix of size $n times n$. The $k$th elementary elimination matrix for it is defined as

$
M_k = mat(
  1, dots, 0, 0, 0, dots, 0;
  dots.v, dots.down, dots.v, dots.v, dots.v, dots.down, dots.v;
  0, dots, 1, 0, 0, dots, 0;
  0, dots, 0, 1, 0, dots, 0;
  0, dots, 0, -m_(k+1), 1, dots, 0;
  dots.v, dots.down, dots.v, dots.v, dots.v, dots.down, dots.v;
  0, dots, 0, -m_n, 0, dots, 1
)
$

where $m_i = a_(i k)/a_(k k)$.

By applying this elementary elimination matrix $M_1$ on $A$, we can obtain a new matrix with the $a'_(i 1) = 0$ for all $i>1$.
$
M_1 A = mat(
  a_(1 1), a_(1 2), a_(1 3), dots.h, a_(1 n);
  0, a'_(2 2), a'_(2 3), dots.h, a'_(2 n);
  0, a'_(3 2), a'_(3 3), dots.h, a'_(3 n);
  dots.v, dots.down, dots.v, dots.down, dots.v;
  0, a'_(n 2), a'_(n 3), dots.h, a'_(n n)
)
$

For $k=1,2,dots,n$, apply $M_k$ on $A$. We will have an upper triangular matrix.
$
U = M_(n-1) dots.h M_1 A
$

Since $M_k$ is reversible, we have
$
A = L U\
L = M_1^(-1) M_2^(-1) dots.h M_(n-1)^(-1)
$
Elementary elimination matrices have the following properties that making the above process efficient:
1. Its inverse can be computed in $O(n)$ time
   $
   M_k^(-1) = 2I - M_k
   $
2. The multiplication of two elementary matrices can be computed in $O(n)$ time
   $
   M_k M_(k' > k) = M_k + M_(k') - I
   $

== Code: Elementary Elimination Matrix

```julia
A3 = [1 2 2; 4 4 2; 4 6 4]

function elementary_elimination_matrix(A::AbstractMatrix{T}, k::Int) where T
    n = size(A, 1)
    @assert size(A, 2) == n
    # create Elementary Elimination Matrices
    M = Matrix{Float64}(I, n, n)
    for i=k+1:n
        M[i, k] =  -A[i, k] ./ A[k, k]
    end
    return M
end
```

The elementary elimination matrix for the above matrix `A3` eliminating the first column is

```julia
elementary_elimination_matrix(A3, 1)
elementary_elimination_matrix(A3, 1) * A3
```

Verify the property 1

```julia
inv(elementary_elimination_matrix(A3, 1))
```

Verify the property 2

```julia
elementary_elimination_matrix(A3, 2)
inv(elementary_elimination_matrix(A3, 1)) * inv(elementary_elimination_matrix(A3, 2))
```

A naive implementation of elimentary elimination matrix is as follows


```julia
function lufact_naive!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n
    M = Matrix{T}(I, n, n)
    for k=1:n-1
        m = elementary_elimination_matrix(A, k)
        M = M * inv(m)
        A .= m * A
    end
    return M, A
end

lufact_naive!(copy(A3))

@testset "naive LU factorization" begin
    A = [1 2 2; 4 4 2; 4 6 4]
    L, U = lufact_naive!(copy(A))
    @test L * U ≈ A
end
```

The above implementation has time complexity $O(n^4)$ since we did not use the sparsity of elimentary elimination matrix. A better implementation that gives $O(n^3)$ time complexity is as follows.

