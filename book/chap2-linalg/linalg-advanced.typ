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