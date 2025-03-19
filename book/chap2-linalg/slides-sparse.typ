#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#set math.mat(row-gap: 0.1em, column-gap: 0.7em)

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
  title: [Sparse Matrices and Dominant Eigenvalues],
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


#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

= Sparse Matrices

== Example 1: The stiffness matrix of a spring chain

$
   mat(
    -C, C, 0, dots.h, 0;
    C, -2C, C, dots.h, 0;
    0, C, -2C, dots.h, 0;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    0, 0, 0, C, -C
  )
vec(u_1, u_2, u_3, dots.v, u_n) = -M omega^2 vec(u_1, u_2, u_3, dots.v, u_n)
$

- _Remark_: Many operators in linear differential equations are sparse.

== Example 2: The adjacency matrix of a graph

#box(text(16pt)[```julia
julia> using Graphs; adjacency_matrix(random_regular_graph(1000, 3))
1000×1000 SparseMatrixCSC{Int64, Int64} with 3000 stored entries:
⎡⣠⣺⠳⠖⡳⢏⣰⣶⣶⡍⠖⠯⢽⣗⣸⢜⣯⣷⢾⣂⢰⣇⣫⡮⠟⢶⡌⠐⎤
⎢⢹⠆⡮⣫⡫⣯⣿⣺⢕⢢⣲⢴⠥⣲⡍⠻⢿⠞⠯⣶⡪⣻⢜⡹⢟⣏⢇⢋⎥
⎢⡽⢎⡯⣮⣺⡾⣤⡉⢏⢿⡌⡿⡯⢵⡚⡙⡞⣯⢉⣰⢷⣸⣻⣲⡾⠛⣛⢃⎥
⎢⢰⣾⣻⣻⡄⠻⣞⡹⢯⣽⢸⡃⡿⠽⣯⣻⢅⡨⢞⣿⡣⣣⣾⣺⣛⡯⢿⢵⎥
⎢⡜⠿⠱⣑⣯⣕⣏⣷⣎⠙⣼⡽⣫⡽⣖⡣⢯⡽⣫⣞⠺⠛⣿⣱⠯⡪⡿⡶⎥
⎢⡼⡅⢘⣞⣦⡭⠶⠲⣖⡿⣮⢛⣕⣹⡡⡽⠺⢽⣾⣇⡋⣴⡫⣚⡽⡹⡲⡫⎥
⎢⢷⢷⢡⣣⢏⣏⣟⡏⣏⡾⣕⣹⡥⠏⣆⣗⡧⠜⣭⢷⢗⢚⣍⣶⠫⣩⡛⡕⎥
⎢⣒⢞⣧⡉⣞⠨⣯⣻⠼⡹⣅⡮⢬⢽⣪⡾⢗⣖⣹⣭⡮⣳⠼⣕⢾⡛⣽⡣⎥
⎢⢯⣿⣻⠗⡾⣭⡁⡱⣏⡷⣞⣆⣉⠏⢹⢵⣪⡺⡞⢶⡼⣅⢟⡻⡹⡚⢺⠄⎥
⎢⠺⢳⢫⣧⢃⣰⣾⣵⣫⢾⠾⢿⢧⣟⡗⣾⢺⣍⣏⣽⣑⡜⡏⡿⣷⢩⣭⠌⎥
⎢⠴⢶⣮⣪⣙⣳⠭⣪⣾⠂⢋⣬⣹⢑⢮⣫⠖⢯⣑⠼⣏⡹⢗⢯⢣⣗⣉⢧⎥
⎢⡫⡾⣖⡱⢻⣺⣺⣻⢟⣻⣫⢪⢣⣽⢖⢧⣿⡱⣯⡭⡽⣕⠡⠂⣾⣞⣅⡕⎥
⎢⢻⣅⡿⢵⣾⠋⡿⡼⡫⡣⣗⡫⡏⣢⣾⠳⣳⠪⡝⣛⢭⢶⣺⢿⣊⡸⠓⡏⎥
⎣⢂⠉⡭⢑⠿⢘⢟⣗⢻⡯⡼⡪⢟⠬⠷⡻⠚⠖⡃⠟⠧⣜⢅⠽⡽⠤⡤⡫⎦
```
])


== Example 3: Quantum physics
#box(text(16pt)[```julia
julia> using Yao; Yao.mat(EasyBuild.heisenberg(20))
1048576×1048576 SparseMatrixCSC{ComplexF64, Int64} with 11164824 stored entries:
⎡⠻⣦⣄⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⢙⡻⣮⡳⡄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠙⠮⢻⣶⡄⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠠⣄⠉⠻⣦⣀⠙⠦⠀⠀⠀⣄⠀⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠈⠳⣄⠘⢿⣷⡰⣄⠀⠀⠈⠳⣄⠀⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠈⠃⠐⢮⡻⣮⣇⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠈⠳⣄⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠿⣧⠀⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠈⠳⣄⎥
⎢⠙⢦⡀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢻⣶⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠙⢦⡀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⢹⡻⣮⡳⠄⢠⡀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠙⠎⢿⣷⡄⠙⢦⡀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠙⠀⠀⠀⠲⣄⠉⠻⣦⣀⠙⠂⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣄⠘⠿⣧⡲⣄⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠘⢮⡻⣮⣅⠀⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠙⠻⣦⎦
```
])

== The essential operations on sparse matrices

- #highlight([Matrix-vector multiplication: $A v$ (essential)])
- Solving linear systems: $A x = b$
- Eigenvalues and eigenvectors: $A v = lambda v$, used in graph spectral theory, quantum physics, etc.
- Expmv: $e^A v$, used in time-evolution simulation of a quantum system
#v(30pt)
- _Remark_: Usually, the above operations does not require the explicit use of the `getindex` function (or `A[i, j]` operator) on a sparse matrix. The matrix-vector multiplication is the essential operation.

== COOrdinate (COO) format

The coordinate format means storing nonzero matrix elements into triples of row index, column index, and value:

$
  &(i_1, j_1, v_1)\
  &(i_2, j_2, v_2)\
  &dots\
  &(i_k, j_k, v_k)
$

To store a sparse matrix $A$ in COO format, we only need to store $"nnz"(A)$ triples, where $"nnz"(A)$ is the number of nonzero elements in $A$. Julia does not have a native COO data type, so we implement a COO matrix from scratch by implementing the #link("https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array")[`AbstractArray`] interface:
- `size`: return the size of the matrix
- `getindex`: return the element at the given index

== Implementing the COO format

#box(text(16pt)[```julia
using LinearAlgebra

struct COOMatrix{Tv, Ti} <: AbstractArray{Tv, 2}   # Julia does not have a COO data type
    m::Ti                # number of rows
    n::Ti                # number of columns
    colval::Vector{Ti}   # column indices
    rowval::Vector{Ti}   # row indices
    nzval::Vector{Tv}    # values
    function COOMatrix(m::Ti, n::Ti, colval::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        @assert length(colval) == length(rowval) == length(nzval)
        new{Tv, Ti}(m, n, colval, rowval, nzval)
    end
end
```
])

#box(text(16pt)[```julia
Base.size(coo::COOMatrix) = (coo.m, coo.n)
Base.size(coo::COOMatrix, i::Int) = getindex((coo.m, coo.n), i)
nnz(coo::COOMatrix) = length(coo.nzval)

function Base.getindex(coo::COOMatrix{Tv}, i::Integer, j::Integer) where Tv
    @boundscheck checkbounds(coo, i, j)
    v = zero(Tv)
    for (i2, j2, v2) in zip(coo.rowval, coo.colval, coo.nzval)
        # If we find a matching position
        if i == i2 && j == j2
            v += v2  # accumulate the value, since repeated indices are allowed.
        end
    end
    return v
end
```
Q: What is the time complexity to get an element?
])

== Matrix vector multiplication
#box(text(16pt)[```julia
function LinearAlgebra.mul!(y::AbstractVector{T}, A::COOMatrix, x::AbstractVector) where T
    @assert size(A, 2) == length(x) "Dimension mismatch"
    @assert size(A, 1) == length(y) "Dimension mismatch"
    
    fill!(y, zero(T))
    # Accumulate contributions from each non-zero element
    for (i, j, v) in zip(A.rowval, A.colval, A.nzval)
        y[i] += v * x[j]
    end
    return y
end
```

Q: What is the time complexity of the above code?
])

== Matrix matrix multiplication
#box(text(16pt)[```julia
function Base.:*(A::COOMatrix{T1}, B::COOMatrix{T2}) where {T1, T2}
    @assert size(A, 2) == size(B, 1)
    rowval = Int[]
    colval = Int[]
    nzval = promote_type(T1, T2)[]  # Promote types to handle mixed precision
    
    for (i, j, v) in zip(A.rowval, A.colval, A.nzval)
        for (i2, j2, v2) in zip(B.rowval, B.colval, B.nzval)
            if j == i2
                push!(rowval, i)
                push!(colval, j2)
                push!(nzval, v * v2)  # Multiply the values
            end
        end
    end
    return COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval)
end
```
Q: What is the time complexity of the above code?
])

In the following, we use the COO format to store the stiffness matrix of a spring chain and test its performance.
$
mat(
  -C, C, 0, dots.h, 0;
  C, -2C, C, dots.h, 0;
  0, C, -2C, dots.h, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, C, -C
)
$

== Performance test

#box(text(16pt)[```julia
using BenchmarkTools

stiffmatrix(n::Int, C) = COOMatrix(n, n, [1:n-1; 1:n; 2:n], [2:n; 1:n; 1:n-1], [C*ones(n-1); -C ; -2C*ones(n-2); -C; C*ones(n-1)])

matrix = stiffmatrix(10000, 1.0)
x = randn(size(matrix, 2))
@btime matrix * x         # 27.292 μs
@btime matrix * matrix    # 800.915 ms
```
])

== Compressed Sparse Column (CSC) format
#highlight([CSC is the most widely used format in Julia's `SparseArrays` standard library.])

In CSC format, a sparse matrix is represented by three arrays:
- `colptr`: An array of column pointers (length n+1)
- `rowval`: An array of row indices for each non-zero element
- `nzval`: An array of non-zero values

#figure(canvas({
  import draw: *
  let dx = 0.8
  let dy = 1.5
  let s(it) = text(12pt, it)
  let boxed(loc, text) = {
    rect(loc.map(x => x - dx/2), loc.map(x=> x + dx/2), stroke: black)
    content(loc, text)
  }
  let rowval = (2, 3, 1, 4, 3, 4)
  let nzval = (1, 2, 3, 4, 5, 6)
  let colptr = (1, 3, 5, 5, 7)

  content((-2, dy), s[colptr])
  for (i, v) in colptr.enumerate() {
    let x = (v - 1.5) * dx
    content((x, dy), s[#v], name: str(i))
    line((x, dy - 0.3), (x, 0.5), mark: (end: "straight"))
  }
  content((dx/2, dy/2), s[$j=1$])
  content((2.5 * dx, dy/2), s[$j=2$])
  content((4.5 * dx, dy/2), s[$j=4$])

  content((-2, 0), s[rowval])
  for (i, v) in rowval.enumerate() {
    boxed((i * dx, 0), s[#v])
  }

  content((-2, -dy), s[nzval])
  for (i, v) in nzval.enumerate() {
    boxed((i * dx, -dy), s[#v])
  }

  content((7, -0.5), text(16pt)[$mat(dot,3,dot, dot;1, dot, dot, dot;2, dot, dot, 5;dot, 4, dot,  6;dot, dot, dot, dot)$])
}))

== Access a column
- `rowval[colptr[j]:colptr[j+1]-1]`, the row indices of the nonzero elements in the $j$-th column,
- `nzval[colptr[j]:colptr[j+1]-1]`, the nonzero values in the $j$-th column.

== Construct a CSC matrix
#box(text(16pt)[```julia
using SparseArrays
# the sparse matrix in the above figure.
sparse([2, 3, 1, 4, 3, 4], [1, 1, 2, 2, 4, 4], [1, 2, 3, 4, 5, 6], 5, 4)
# Output: 5×4 SparseMatrixCSC{Int64, Int64} with 6 stored entries:
#  ⋅  3  ⋅  ⋅
#  1  ⋅  ⋅  ⋅
#  2  ⋅  ⋅  5
#  ⋅  4  ⋅  6
#  ⋅  ⋅  ⋅  ⋅
```
])

== Implement our own CSC matrix
#box(text(16pt)[```julia
struct CSCMatrix{Tv,Ti} <: AbstractMatrix{Tv}
    m::Int              # Number of rows
    n::Int              # Number of columns
    colptr::Vector{Ti}  # Column pointers (length n+1)
    rowval::Vector{Ti}  # Row indices of non-zero elements
    nzval::Vector{Tv}   # Values of non-zero elements
    function CSCMatrix(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        @assert length(colptr) == n + 1
        @assert length(rowval) == length(nzval) == colptr[end] - 1
        new{Tv, Ti}(m, n, colptr, rowval, nzval)
    end
end
```])

== Get an element
#box(text(16pt)[```julia
function Base.getindex(A::CSCMatrix{T}, i::Int, j::Int) where T
    # Check if indices are within bounds
    @boundscheck checkbounds(A, i, j)
    # Search for the row index i in column j
    for k in nzrange(A, j)
        if A.rowval[k] == i
            return A.nzval[k]
        end
    end
    # Return zero if element not found (sparse matrices return zero for missing elements)
    return zero(T)
end

# Return the range of indices in nzval/rowval arrays for column j
nzrange(A::CSCMatrix, j::Int) = A.colptr[j]:A.colptr[j+1]-1
```
])

The row indices and values of nonzero elements in the 3rd column of `cscm` can be obtained by
#box(text(16pt)[```julia
rows3 = cscm.rowval[cscm.colptr[3]:cscm.colptr[4]-1]
val3 = cscm.nzval[cscm.colptr[3]:cscm.colptr[4]-1]
cscm.rowval[nzrange(cscm, 3)] # equivalent to the first approach
```
])

== Matrix-vector multiplication
#box(text(16pt)[```julia
function LinearAlgebra.mul!(y::AbstractVector{T}, A::CSCMatrix,x::AbstractVector{T}) where T
    fill!(y, zero(T))
    
    # Loop through each column of A
    for j in 1:size(A, 2)
        # For each nonzero element in column j
        for k in nzrange(A, j)
            y[A.rowval[k]] += A.nzval[k] * x[j]
        end
    end
    return y
end
```
Q: What is the time complexity of the above code?
])

#box(text(14pt)[```julia
function Base.:*(A::CSCMatrix{T1}, B::CSCMatrix{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    @assert size(A, 2) == size(B, 1)
    
    rowval, colval, nzval = Int[], Int[], T[]
    for j2 in 1:size(B, 2)  # enumerate the columns of B
        for k2 in nzrange(B, j2)  # enumerate the rows of B
            v2 = B.nzval[k2]
            i2 = B.rowval[k2]
            
            # For each nonzero element in column i2 of A
            for k1 in nzrange(A, i2)  # enumerate the rows of A
                push!(rowval, A.rowval[k1])
                push!(colval, j2)
                push!(nzval, A.nzval[k1] * v2)
            end
        end
    end
    return CSCMatrix(COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval))
end
```
Q: What is the time complexity of the above code?
])

== Performance test
#box(text(16pt)[```julia
@btime cscm * x      # 37.042 μs
@btime cscm * cscm    # 3.349 ms
```
])

- Time to multiply two CSC matrices $A$ and $B$: $O(op("nnz")(A)op("nnz")(B)\/n)$
- Time to multiply two COO matrices: $O(op("nnz")(A)op("nnz")(B))$

The implementation in `SparseArrays` is even more efficient.
#box(text(16pt)[```julia
using SparseArrays
sp = sparse(matrix.rowval, matrix.colval, matrix.nzval, matrix.m, matrix.n)
@btime sp * x     # 26.917 μs
@btime sp * sp    # 255.208 μs
```
])

= Dominant eigenvalue problem

== Solving linear equations

Routines in #link("https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl")[`IterativeSolvers.jl`]:

#table(columns: 2,
  table.header([*Method*], [*When to use it*]),
  [Conjugate Gradients], [Best choice for symmetric, positive-definite matrices],
  [MINRES], [For symmetric, indefinite matrices],
  [GMRES], [For nonsymmetric matrices when a good preconditioner is available],
  [IDR(s)], [For nonsymmetric, strongly indefinite problems without a good preconditioner],
  [BiCGStab(l)], [Otherwise for nonsymmetric problems]
)

== Example:

#box(text(16pt)[```julia
using IterativeSolvers, SparseArrays, LinearAlgebra
A = sprandn(10000, 10000, 0.001) + Diagonal(1:10000)
b = randn(10000)

x = gmres(A, b; reltol=1e-10)
norm(A * x - b)  # 9.99579999231606e-9
```])

- _Remark_: The `gmres` function uses the _Arnoldi process_ to solve the problem.

== Dominant eigenvalue problem

Given a matrix $A in RR^(n times n)$, the dominant eigenvalue problem is to find the largest (or smallest) eigenvalue $lambda_1$ and its corresponding eigenvector $x_1$:

$ min_(x_1) lambda_1 quad "s.t." A x_1 = lambda_1 x_1. $

== Power method
The power method is a simple iterative algorithm to solve the dominant eigenvalue problem. The algorithm starts with a random vector $v_0$ and repeatedly multiplies it with the matrix $A$.

$ v_k = A^k v_0 $

By representing the initial vector $v_0$ as a linear combination of eigenvectors of $A$, i.e. $v_0 = sum_(i=1)^n c_i x_i$, we have

$ v_k = sum_(i=1)^n lambda_i^k c_i x_i $

where $lambda_1 > lambda_2 >= dots.h >= lambda_n$ are the eigenvalues of $A$ and $x_i$ are the corresponding eigenvectors. The power method converges to the eigenvector corresponding to the largest eigenvalue as $k -> infinity$. The rate of convergence is determined by $|lambda_2/lambda_1|^k$.

== Implement the power method
#box(text(16pt)[```julia
function power_method(A::AbstractMatrix{T}, n::Int) where T
    n = size(A, 2)
    x = normalize!(randn(n))
    for i=1:n
        x = A * x
        normalize!(x)
    end
    return x' * A * x', x
end
```
])

- _Remark_: By inverting the sign, $A -> -A$, we can use the same method to obtain the smallest eigenvalue.

== KrylovKit.jl
The Julia package #link("https://github.com/Jutho/KrylovKit.jl")[`KrylovKit.jl`] contains many Krylov space based algorithms.
`KrylovKit.jl` accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.
The high level interface of KrylovKit is provided by the following functions:
- `linsolve`: solve linear systems
- `eigsolve`: find a few eigenvalues and corresponding eigenvectors
- `geneigsolve`: find a few generalized eigenvalues and corresponding vectors
- `svdsolve`: find a few singular values and corresponding left and right singular vectors
- `exponentiate`: apply the exponential of a linear map to a vector
- `expintegrator`: #link("https://en.wikipedia.org/wiki/Exponential_integrator")[exponential integrator]
    for a linear non-homogeneous ODE, computes a linear combination of the $phi_j$ functions which generalize $phi_0(z) = exp(z)$.

== Example: Solve the slowest mode of the spring chain.

We use the `KrylovKit.eigsolve` function. This function accepts a linear map, an initial vector, the number of eigenvalues to compute, and the target eigenvalue type.

#box(text(16pt)[
```julia
julia> using KrylovKit

julia> eigsolve(sp, randn(size(sp, 1)), 1, :SR)
([-3.999996868663288], ..., ConvergenceInfo: no converged values after 100 iterations and 1218 applications of the linear map;
norms of residuals are given by (6.611549984040687e-5,).
```
])
- `:SR` means the "smallest" real part of the eigenvalue.
- The output contains three parts, the eigenvalues, the eigenvectors, and the convergence information.
- _Remark_: Since this matrix is real symmetric, all eigenvalues and all eigenvectors are real. It calls into the Lanczos algorithm.

== Correct the target eigenvalue type
Let us correct the target eigenvalue type using the `EigSorter`.

#box(text(16pt)[
```julia
julia> eigsolve(sp, randn(size(sp, 1)), 1, EigSorter(abs; rev=false))
([-3.148596175606358e-6], ..., ConvergenceInfo: no converged values after 100 iterations and 1218 applications of the linear map;
norms of residuals are given by (8.144476258164312e-5,).
)
```
])

It produces a value $~-3times 10^(-6)$, which is very close to 0 as expected. We do not know if it is a zero eigenvalue, or just being very small, since the residual is still large.

== Improve the precision
Let us improve the precision by setting a larger tolerance and a larger maximum number of iterations:

#box(text(16pt)[
```julia
julia> eigsolve(sp, randn(size(sp, 1)), 1, EigSorter(abs; rev=false), tol=1e-10, maxiter=5000)
([-4.778184048570348e-8], ..., ConvergenceInfo: no converged values after 5000 iterations and 60018 applications of the linear map;
norms of residuals are given by (1.0713185997288401e-6,).
```
])

Now the residual is much smaller, and the eigenvalue is also two orders of magnitude smaller.
Hence, we are more confident that it is a zero eigenvalue.

== The Krylov subspace method

The Krylov subspace method is the method that `KrylovKit.jl` uses to solve the large scale dominant eigenvalue problem. The input can be any kind of linear map that implements the `mul!` interface, which performs the matrix-vector multiplication. The linear map does not have to be a dense or sparse matrix, but can be any kind of linear operator $A$ satisfying:

$ A (alpha x + beta y) = alpha A(x) + beta A(y). $

== The Krylov subspace method
The Krylov subspace method, such as the Arnoldi and Lanczos algorithms, have much faster convergence speed comparing with the power method.
The key idea is to generate an orthogonal matrix $Q in CC^(n times k)$ with $k << n$, $Q^dagger Q = I$, such that
$ Q^dagger A Q = B, $
and the largest eigenvalue of $B$ best approximates the largest eigenvalue of $A$.
Since $B$ has size $k times k$, the new eigenvalue problem is much easier to solve. The largest eigenvalue of $B$ upper bounds the largest eigenvalue of $A$:
$ lambda_1(B) <= lambda_1(A), $
where $lambda_1(A)$ denotes the largest eigenvalue of $A$. The equality holds if $Q$ is chosen such that $op("span")(Q)$ contains the dominant eigenvectors of $A$.

== Why?
Whenever, we have $B y_1 = lambda_1(B) y_1$, we have $y_1^dagger Q^dagger A Q y_1 = lambda_1(B) y_1^dagger y_1 = lambda_1(A)$. Then, either $Q y_1$ is the largest eigenvector of $A$, or $A$ has an eigenvalue $lambda_1(A)$ such that $lambda_1(A) > lambda_1(B)$.

== The Krylov subspace
The $Q$ can be generated from the *Krylov subspace* generated from a random initial vector $q_1$:
$ cal(K)(A, q_1, k) = op("span"){q_1, A q_1, A^2 q_1, dots, A^(k-1) q_1}. $ <eq:krylov-subspace>
Unlike the power method, the Krylov subspace method generates an orthogonal matrix $Q$ by orthonormalizing the Krylov vectors, rather than just using the last vector. Hence, #highlight([it is strictly better than the power method.])

== The Lanczos algorithm

The Lanczos algorithm is a special case of the Krylov subspace method that designed for #highlight([symmetric linear operators]). It is an iterative process to generate the subspace spanned by the Krylov vectors. The the Lanczos algorithm generates an orthogonal matrix $Q$ such that

$ Q^dagger A Q = T $

where $T$ is a symmetric tridiagonal matrix

$ T = mat(
  alpha_1, beta_1, 0, dots, 0;
  beta_1^*, alpha_2, beta_2, dots, 0;
  0, beta_2^*, alpha_3, dots, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, beta_(k-1)^*, alpha_k
), $
where $Q = (q_1 | q_2 | dots | q_k)$, and $op("span")({q_1, q_2, dots, q_k}) = cal(K)(A, q_1, k)$.

== The Lanczos algorithm
The Lanczos algorithm is basically a Gram-Schmidt orthogonalization process applied to the Krylov subspace:
1. We start with a normalized vector $q_1$ and compute $A q_1$ (the second vector in the Krylov subspace).
2. To find $alpha_1$, we project $A q_1$ onto $q_1$ by computing the inner product:
   $ alpha_1 = q_1^dagger A q_1 $
3. The remainder $r_1 = A q_1 - alpha_1 q_1$ is orthogonal to $q_1$. We set $beta_1 = ||r_1||_2$ and $q_2 = r_1\/beta_1$ (if $beta_1 != 0$).
4. For subsequent steps, we compute $A q_k$ and make it orthogonal to both $q_k$ and $q_(k-1)$ by subtracting their projections:
   $ r_k = A q_k - alpha_k q_k - beta_(k-1) q_(k-1) $

== The Lanczos algorithm
The key insight is that, for symmetric matrices, $r_k$ is automatically orthogonal to $q_1, q_2, dots, q_(k-2)$ due to the properties of the Krylov subspace. This is why we only need to explicitly orthogonalize against the two most recent vectors.

The iteration terminates when $beta_k = 0$, it means the residual vector is zero, indicating that the Krylov subspace has become invariant under $A$. In this case, we have found an exact invariant subspace, and the algorithm terminates. The resulting tridiagonal matrix takes a block-diagonal form:

$ T( "when " beta_j = 0) = mat(
  alpha_1, beta_1, 0, dots, 0;
  beta_1^*, alpha_2, beta_2, dots, 0;
  0, beta_2^*, alpha_3, dots, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, beta_(j-1)^*, alpha_j
), $

== The Lanczos algorithm
This block structure reflects the fact that the Krylov subspace has split into invariant subspaces of $A$.

In practice, the Gram-Schmidt process is numerically unstable, and the orthogonalization process is often replaced by a more numerically stable process, such as the modified Gram-Schmidt process or iterative reorthogonalization.

== Reorthogonalization

In the Lanczos algorithm, numerical errors can accumulate and cause the orthogonality of the basis vectors to deteriorate. Reorthogonalization is a technique to maintain orthogonality among these vectors.

One effective approach uses Householder transformations. Let's understand how this works:

1. Consider linearly independent vectors $r_0, dots, r_(k-1) in CC^n$.

2. For each vector $r_i$, we can construct a Householder matrix $H_i$ that reflects vectors across a hyperplane.

3. When we apply the sequence of Householder matrices $H_0, dots, H_(k-1)$ to the matrix $(r_0|dots|r_(k-1))$, we get:
   $(H_0 dots H_(k-1))^T (r_0|dots|r_(k-1)) = R$
   where $R$ is an upper triangular matrix.

4. If we denote the first $k$ columns of the matrix product $(H_0 dots H_(k-1))$ as $(q_1 | dots | q_k)$, then these vectors $q_1, dots, q_k$ form an orthonormal basis.

== Essential aspects of a professional Lanczos implementation
A professional Lanczos implementation should also consider the following aspects:
1. *Block Lanczos*: Used to compute degenerate eigenvalues. Instead of using a single vector, the block Lanczos method uses a block of $p$ vectors at each iteration. This approach is particularly effective for matrices with clustered or degenerate eigenvalues, as it can capture multiple eigenvectors in the same invariant subspace simultaneously. The algorithm generates a block tridiagonal matrix rather than a simple tridiagonal one.
2. *Restarting*: Used to reduce the memory usage. When the Krylov subspace becomes too large, we can restart the algorithm by compressing the information from the current iteration into a smaller subspace. The technique of _implicit restarting_ allows us to focus on the most relevant part of the spectrum without increasing memory usage. The Implicitly Restarted Lanczos Method (IRLM) combines $m$ steps of the Lanczos process with implicit QR steps to enhance convergence toward desired eigenvalues. A variant of implicit restarting is _thick restarting_, where we keep a few (typically converged) Ritz vectors and restart the Lanczos process with these vectors plus a new starting vector. This approach maintains good approximations while exploring new directions in the Krylov subspace.
These techniques could be found in @Golub2013.

== The Arnoldi algorithm

The Arnoldi algorithm is a generalization of the Lanczos algorithm to non-symmetric linear operators. Given a non-symmetric linear operator $A$ on $RR^n$, the Arnoldi algorithm generates an orthogonal matrix $Q$ such that $Q^T A Q = H$ is a Hessenberg matrix:


$
H = mat(
  h_(11), h_(12), h_(13), dots, h_(1k);
  h_(21), h_(22), h_(23), dots, h_(2k);
  0, h_(32), h_(33), dots, h_(3k);
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, dots, h_(k k)
)
$


That is, $h_(i j) = 0$ for $i>j+1$.

==
```julia
```

In the following example, we use the Arnoldi algorithm to find the eigenvalues of a random matrix.
```julia
```
- _Remark_: Sometimes, the `KrylovKit.jl` returns more than the requested number of eigenvalues. Most of the time, the extra eigenvalues are due to the degeneracy of the eigenvalues. They converge simultaneously.

== Special matrices

Some special matrices have special properties that can be exploited for efficient computation.

=== (Block) Tridiagonal matrices

#figure(table(columns: 2,
    table.header([*Operation*], [*Time complexity*]),
    [*Linear system solving*], [$O(n)$ (Thomas algorithm)],
    [*Matrix inversion*], [$O(n^2)$@Ran2006],
    [*Determinant*], [$O(n)$@Molinari2008],
    [*Eigenvalue problem*], [$O(n^2)$@Dhillon2004 @Sandryhaila2013],
))

= Hands-on
== Hands-on: Implement and improve a simple Lanczos algorithm
1. Run the demo code in folder: `SimpleKrylov/examples` with:
   ```bash
   $ make init-SimpleKrylov
   $ make example-SimpleKrylov
   ```
2. Explain the inconsistency between the results of `SimpleKrylov` and the exact results.
3. Verify the following property for the Laplacian matrix of a graph:
   The number of connected components in the graph is the dimension of the nullspace of the Laplacian and the algebraic multiplicity of the 0 eigenvalue. The Laplacian matrix is defined as $L = D - A$, where $D$ is the diagonal degree matrix and $A$ is the adjacency matrix.

==
#bibliography("refs.bib")