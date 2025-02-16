#import "@preview/cetz:0.2.2": *
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
B[2]          # the second element in linearized representation
B[1:2, 1:2]   # 2×2 submatrix
```

== Map, reduction, broadcasting, filtering and searching

_Broadcasting_ in Julia provides a powerful way to apply functions element-wise across arrays. For example, to compute function $y = sin(x) + cos(3x)$ for each element $x$ in an array, you can use the following syntax:

```julia
x = 0:0.1π:2π
y = sin.(x) .+ cos.(3 .* x)        # Broadcasting
y = map(a -> sin(a) + cos(3a), x)  # The mapping version
```

When you have multiple broadcasting operations in a single expression, Julia performs loop fusion, executing all operations in a single pass without creating intermediate arrays. This often leads to better performance than having separate operations.

Sometimes, you may want to protect an object from broadcasting. You can use `Ref` to prevent broadcasting over an entire object:

```julia
Ref([3,2,1,0]) .* (1:3)  # returns [[3, 2, 1, 0], [6, 4, 2, 0], [9, 6, 3, 0]]
```

_Reduction_ is a common operation in scientific computing. Given a generic vector of size $n$ and element type $T$, $bold(v) in T^n$, left and right folding this vector with a function $f: T times T arrow.r T$ is equivalent to computing
$
f(f(f(v_1, v_2), v_3), ..., v_n),
$
and 
$
f(v_1, f(v_2, f(v_3, ..., f(v_(n-1), v_n))))
$
respectively.

For example, we can use `foldl` and `foldr` as follows:
```julia
foldl((x, y) -> [x, y], [1, 2, 3, 4])  # returns [[[1, 2], 3], 4]
foldr((x, y) -> [x, y], [1, 2, 3, 4])  # returns [1, [2, [3, 4]]]
```
In many cases, the operation is commutative, so the result is the same regardless of the direction of folding. For example, to compute the sum of all elements in an array, you can use the following syntax:

```julia
sum(1:10)
foldl(+, 1:10)
foldr(+, 1:10)
reduce(+, 1:10)
```

They are equivalent to each other in this case. `reduce` does not promise the order of evaluation, but it brings advantage in parallelization.

_Map-reduce_ is an even more powerful operation that applies a function to each element of an array and then reduces the result. For example, to compute the squared norm of a vector, you can use the following syntax:
```julia
sum(abs2, 1:10)
mapreduce(abs2, +, 1:10)
```
The first argument of `mapreduce` is the function applied to each element, the second argument is the reduction operation, and the third argument is the array.
The whole process does not create intermediate arrays.

_Filtering_ is another common operation in scientific computing. For example, to filter the even elements in an array, you can use the following syntax:
```julia
filter(iseven, 1:10)  # returns [2, 4, 6, 8, 10]
```

_Searching_ specific element(s) from an array can be achieved by `findfirst`, `findlast`, and `findall`:
```julia
findfirst(iseven, 1:10)  # returns 2
findlast(iseven, 1:10)   # returns 10
findall(iseven, 1:10)   # returns [2, 4, 6, 8, 10]
```

== High dimensional array indexing

Arrays are stored as vectors in memory, either in row-major or column-major order. If a matrix is stored in row-major order, the elements are stored in the order on the left panel:
#align(center, canvas({
  import draw: *
  let dx = 1
  let dy = 0.6
  content((0, 0), text(14pt)[$ mat(a_(1 1), a_(1 2), a_(1 3); a_(2 1), a_(2 2), a_(2 3); a_(3 1), a_(3 2), a_(3 3)) $])
  line((-dx, dy), (-dx, -dy), (0, dy), (0, -dy), (dx, dy), (dx, -dy), mark: (end: "straight"))
  content((0, -1.5), text(12pt)[Column-major order])
  content((0, -2), text(12pt)[(Julia, Fortran)])
  set-origin((5, 0))
  content((0, 0), text(14pt)[$ mat(a_(1 1), a_(1 2), a_(1 3); a_(2 1), a_(2 2), a_(2 3); a_(3 1), a_(3 2), a_(3 3)) $])
  line((-dx, dy), (dx, dy), (-dx, 0), (dx, 0), (-dx, -dy), (dx, -dy), mark: (end: "straight"))
  content((0, -1.5), text(12pt)[Row-major order])
  content((0, -2), text(12pt)[(C, Python)])
}))

Given a matrix `A` of size `(m, n)` stored in the column-major order, the row stride is $1$, while the column stride is $m$. It means the distance in memory between `A[i,j]` and `A[i+1,j]` is $1$, while the distance between `A[i,j]` and `A[i,j+1]` is $m$. When extends to higher dimension, we use strides to describe the distance between elements in each dimension.
```julia
A = randn(3, 4, 5)
st = strides(A)  # returns (1, 3, 12)
```
Strides can be used to efficiently access elements in an array. For example, to access the element `A[2,3,2]`, we can use
```julia
ids = [2, 3, 2]

A[1 + st[1] * (ids[1]-1) + st[2] * (ids[2]-1) + st[3] * (ids[3]-1)]
A[mapreduce(i -> st[i] * (ids[i]-1), +, 1:ndims(A), init=1)]
```

In Julia, linear indices and cartesian indices can be converted to each other by `LinearIndices` and `CartesianIndices`:
```julia
inds = LinearIndices(A)
inds[2,3,2]  # returns 20
inds = CartesianIndices(A)
inds[20]     # returns CartesianIndex(2, 3, 2)
```

The memory layout significantly affect the performance of array operations. Consider two implementations of the Frobenius norm:

```julia
# Row-major traversal (slower)
function frobenius_norm(A::AbstractMatrix)
    s = zero(eltype(A))  # zero element of the same type as the array
    @inbounds for i in 1:size(A, 1)  # remove the bounds check
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

The column-major version is typically much faster due to better cache utilization.
```julia
julia> using BenchmarkTools

julia> A = randn(3000, 3000);

julia> @btime frobenius_norm($A);
  44.408 ms (0 allocations: 0 bytes)

julia> @btime frobenius_norm_colmajor($A);
  10.235 ms (0 allocations: 0 bytes)
```

We can see by simply changing the order of the loop, the performance is improved by more than 2 times. This is because the memory access pattern is more cache-friendly.
As shown in the figure @fig:memory-access, the cache is a small and fast memory that is located on or close to the CPU. `L3` cache is the largest and slowest, `L1` cache is the smallest and fastest. When the data is loaded from the main memory to the cache, the data is loaded in chunks.
When CPU accesses the data, if the data is in the cache, it is called a cache hit, otherwise it is called a cache miss. The cache hit rate is a key factor that affects the performance of the program.
When accessing the matrix in the column-major order in Julia, the stride is 1, so the cache hit rate is the highest.

#figure(canvas({
  import draw: *
  let dx = 0.5
  let dy = 1.2
  let s(it) = text(11pt, it)
  content((-2, dx/2), [Main Memory])
  for i in range(20){
    rect((dx *i, 0), (dx * i + dx, dx), name: "m" + str(i), fill: if (2 < i and i < 8) { red } else { white })
  }
  content((-2, dx/2 - dy), [Caches (L3, L2, L1)])
  bezier("m4.north", "m5.north", (rel: (dx/2, 1)), mark: (end: "straight"), name: "s1")
  content((rel: (-2, 0.3), to: "s1.mid"), s[Small stride, high hit rate])

  bezier("m4.north", "m11.north", (rel: (3 * dx + dx/2, 2)), mark: (end: "straight"), name: "s2")
  content((rel: (0, 0.3), to: "s2.mid"), s[Large stride, low hit rate])
  for i in range(5){
    rect((dx *i, -dy), (dx * i + dx, dx - dy), name: "c" + str(i))
  }
  line("m5.south", "c2.north", mark: (end: "straight"), name: "l1")
  content((rel: (2.5, 0), to: "l1.mid"), s[High Latency, chunk-wise])
  content((-2, dx/2 - 2*dy), [CPU Registers])
  for i in range(1){
    rect((dx *i, -2*dy), (dx * i + dx, dx - 2*dy), name: "r" + str(i))
  }
  line("c1.south", "r0.north", mark: (end: "straight"), name: "l2")
  content((rel: (1.2, 0), to: "l2.mid"), s[Low Latency])
}), caption: [Memory access patterns. The data reading from the main memory can have high latency. When accessing data in the memory, the data is automatically loaded into the caches, which have lower latency. The data in the caches are further loaded into the CPU registers, which have the lowest latency.]) <fig:memory-access>

== BLAS and LAPACK

BLAS and LAPACK are the backends of linear algebra operations in many languages, including Julia.
- BLAS (Basic Linear Algebra Subprograms) is a collection of routines that perform basic vector and matrix operations, such as addition, subtraction, multiplication, and division.
- LAPACK (Linear Algebra PACKage) is a library of routines for solving systems of linear equations, least squares problems, eigenvalue problems, and singular value problems. It is built on top of BLAS.

In Julia, you can call BLAS and LAPACK routines directly by using the `LinearAlgebra.BLAS` and `LinearAlgebra.LAPACK` modules. For example, to compute the 2-norm of a vector at odd indices, you can use the following syntax:
```julia
julia> using LinearAlgebra

julia> BLAS.nrm2(4, fill(1.0, 8), 2)  # number of elements is 4, stride is 2
2.0
```
These low-level routines are not easy to use. Julia `LinearAlgebra` module provides a high-level interface to BLAS and LAPACK routines. For example, to compute the 2-norm of a vector, you can use `LinearAlgebra.norm` instead, to compute the matrix multiplication, you can use "`*`" operation instead.

The matrix multiplication in BLAS can fully utilize the modern CPUs, which provides a golden standard for the measuring the performance of a computing device. The performance is usually measured by the number of *floating point operations per second* (FLOPS).
The floating point operations include addition, subtraction, multiplication and division. The FLOPS of a computing device can be related to multiple factors, such as the clock frequency, the number of cores, the number of instructions per cycle, and the number of floating point units. The simplest way to measure the FLOPS is to benchmark the speed of matrix multiplication:

```julia
julia> @btime $A * $A
  2.967 ms (3 allocations: 7.63 MiB)
```

Since the number of FLOPS in a $n times n times n$ matrix multiplication is $2n^3$ (half of the operations are additions), the FLOPS can be calculated as: $2 times 1000^3 / (2.967 times 10^(-3)) approx 674 "GFLOPS"$.

Ideally, the performance of matrix multiplication in all programming languages (Julia, Python, C, Matlab, etc.) using the same BLAS library should be the same. If the matrix multiplication does not reach the expected performance, you can
1. Check the vendor's BLAS library
  ```julia
  julia> using LinearAlgebra

  julia> BLAS.get_config()
  LinearAlgebra.BLAS.LBTConfig
  Libraries: 
  └ [ILP64] libopenblas64_.so
  ```
  Here, we use the `libopenblas64_.so` library, which is the OpenBLAS library. For Intel CPUs, using the MKL library can achieve better performance.

2. Check if the multi-threading is enabled:
  ```julia
  julia> BLAS.get_num_threads()
  16
  ```
  If the number of threads is not the maximum, you can set the number of threads manually:
  ```julia
  julia> BLAS.set_num_threads(32)
  ```
  A special reminder is the number of threads used by BLAS is not the same as the number of threads used by Julia. In Julia, you can use the following command to get the number of threads:
  ```julia
  julia> Base.Threads.nthreads()
  1
  ```
  This may be different from `BLAS.get_num_threads()`.

LAPACK is also a low-level library, e.g. to compute the singular value decomposition of a matrix, you can use `LAPACK.gesvd!` that takes 3 arguments. Alternatively, you can use `LinearAlgebra.svd` that takes 1 argument to make life easier.
```julia
julia> U, S, V = LAPACK.gesvd!('O', 'S', copy(A));

julia> results = svd(A);
```

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

using CairoMakie
scatter(vec(getindex.(mesh2, 1)), vec(getindex.(mesh2, 2)))  # scatter(x, y)
```
Here, we use the `scatter` function from the #link("https://github.com/MakieOrg/Makie.jl")[`CairoMakie`] package to visualize the triangular lattice, which takes two vectors representing the $x$ and $y$ coordinates of the points as input.
The `CairoMakie` package is the default data visualization method in the rest of the book.
#image("images/triangle.svg", width: 400pt)