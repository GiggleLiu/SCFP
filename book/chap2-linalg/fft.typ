#import "../book.typ": book-page

#show: book-page.with(title: "Fast Fourier Transform")

= Fast Fourier Transform

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

== The Cooley–Tukey's Fast Fourier transformation (FFT)

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
