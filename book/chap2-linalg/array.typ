#import "../book.typ": book-page

#show: book-page.with(title: "Manipulating array")

= Manipulating arrays

== Array initialization

Arrays in Julia can be initialized in several ways:

```julia
# Basic array creation
A = [1, 2, 3]                         # vector
B = [1 2 3; 4 5 6; 7 8 9]             # matrix
zero_vector = zeros(3)                # zero vector
rand_vector = randn(Float32, 3, 3)    # random normal distribution
step_vector = collect(1:3)            # collect from range
uninitialized_vector = Vector{Int}(undef, 3)  # uninitialized vector
```

Unlike C, Python, and R, Julia array indexing starts from 1. This design choice aligns with mathematical notation and many scientific computing conventions.
```julia
A = [1, 2, 3]
A[1]      # first element
A[end]    # last element
A[1:2]    # first two elements
A[2:-1:1] # first two elements in reverse order

B = [1 2 3; 4 5 6; 7 8 9]
B[1:2]        # first two elements (column-major)
B[1:2, 1:2]   # 2×2 submatrix
```

== Map, reduction, broadcasting, filtering and searching

Broadcasting in Julia provides a powerful way to apply functions element-wise across arrays. The dot syntax (`.`) indicates broadcasting:

```julia
x = 0:0.1π:2π
y = sin.(x) .+ cos.(3 .* x)
```

#box(stroke: 1pt, inset: 10pt)[
  Broadcasting performs loop fusion, executing all operations in a single pass without creating intermediate arrays. This often leads to better performance than explicit loops.
]

=== Protecting Objects from Broadcasting

Use `Ref` to prevent broadcasting over an entire object:

```julia
Ref([3,2,1,0]) .* (1:3)  # Vector treated as single value
```

== High dimensional array indexing and strides

Column-Major Array Storage

Julia stores arrays in column-major order, which can significantly impact performance.

#figure(
  image("images/colmajor.png", width: 200pt),
  caption: [Column-major memory layout]
)

Consider two implementations of the Frobenius norm:

```julia
# Row-major traversal (slower)
function frobenius_norm(A::AbstractMatrix)
    s = zero(eltype(A))
    @inbounds for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            s += A[i, j]^2
        end
    end
    return sqrt(s)
end

# Column-major traversal (faster)
function frobenius_norm_colmajor(A::AbstractMatrix)
    s = zero(eltype(A))
    @inbounds for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            s += A[i, j]^2
        end
    end
    return sqrt(s)
end
```

#box(stroke: 1pt, inset: 10pt)[
  The column-major version is typically 2x faster due to better cache utilization. When accessing array elements, following the memory layout improves performance.
]

== Example: Triangular Lattice Generation

Here's how to create a triangular lattice using two different approaches:

```julia
b1 = [1, 0]
b2 = [0.5, sqrt(3)/2]
n = 5

# List comprehension approach
mesh1 = [i * b1 + j * b2 for i in 1:n, j in 1:n]

# Broadcasting approach
mesh2 = (1:n) .* Ref(b1) .+ (1:n)' .* Ref(b2)
```

// #figure(
//   image("images/triangle.svg", width: 200pt),
//   caption: [Triangular lattice visualization]
// )

== Storage and performance

Matrix multiplication is fundamental in scientific computing. Julia's built-in `*` operator leverages optimized BLAS libraries:

```julia
A = randn(1000, 1000)
B = similar(A)

# Benchmark results show ~165 GFLOPS on typical hardware
@benchmark $A * $B
```

#box(stroke: 1pt, inset: 10pt)[
  For an n×n matrix multiplication, the operation count is 2n³. Modern CPUs can achieve hundreds of GFLOPS through vectorization and multiple cores.
]