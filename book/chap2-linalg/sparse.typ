#import "../book.typ": book-page
#import "@preview/cetz:0.2.2": *

#show: book-page.with(title: "Sparse Matrices and Graphs")
#let exampleblock(it) = block(fill: rgb("#ffffff"), inset: 1em, radius: 4pt, width: 100%, stroke: black, it)
#set math.equation(numbering: "(1)")

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

= Sparse Matrices and Graphs

== Sparse Matrices

Sparse matrices are ubiquitous in scientific computing. They arise naturally in many applications where most elements are zero, allowing for significant memory savings and computational efficiency. This section considers how to efficiently store and manipulate sparse matrices.

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

```julia
using LinearAlgebra

struct COOMatrix{Tv, Ti} <: AbstractArray{Tv, 2}   # Julia does not have a COO data type
    m::Ti                # number of rows
    n::Ti                # number of columns
    colval::Vector{Ti}   # column indices
    rowval::Vector{Ti}   # row indices
    nzval::Vector{Tv}    # values
    function COOMatrix(m::Ti, n::Ti, colval::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        # Check that all arrays have the same length
        @assert length(colval) == length(rowval) == length(nzval)
        # Create a new COOMatrix with the given parameters
        new{Tv, Ti}(m, n, colval, rowval, nzval)
    end
end

# Return the dimensions of the matrix
Base.size(coo::COOMatrix) = (coo.m, coo.n)
# Return a specific dimension of the matrix
Base.size(coo::COOMatrix, i::Int) = getindex((coo.m, coo.n), i)
# Return the number of non-zero elements in the matrix
nnz(coo::COOMatrix) = length(coo.nzval)

# Implement indexing for COO matrix (A[i,j])
function Base.getindex(coo::COOMatrix{Tv}, i::Integer, j::Integer) where Tv
    # Check if indices are within bounds
    @boundscheck checkbounds(coo, i, j)
    # Initialize return value to zero
    v = zero(Tv)
    # Iterate through all stored elements
    for (i2, j2, v2) in zip(coo.rowval, coo.colval, coo.nzval)
        # If we find a matching position
        if i == i2 && j == j2
            # Add the value (accumulate in case of duplicates)
            v += v2  # accumulate the value, since repeated indices are allowed.
        end
    end
    # Return the accumulated value (or zero if no matches found)
    return v
end
```

The indexing of a COO matrix is slow, making it primarily useful as an input format rather than for computation.
Unless we sort the nonzero elements and remove the duplicate indices, the indexing operation requires $O(op("nnz")(A))$ operations.
In the following, we implement a more efficient version of the matrix-vector and matrix-matrix multiplication for COO matrices.

```julia
# Matrix-vector multiplication using mul!
function LinearAlgebra.mul!(y::AbstractVector{T}, A::COOMatrix, x::AbstractVector) where T
    # Check dimensions
    @assert size(A, 2) == length(x) "Dimension mismatch"
    @assert size(A, 1) == length(y) "Dimension mismatch"
    
    # Zero out the result vector
    fill!(y, zero(T))
    
    # Accumulate contributions from each non-zero element
    for (i, j, v) in zip(A.rowval, A.colval, A.nzval)
        y[i] += v * x[j]
    end
    return y
end

function Base.:*(A::COOMatrix{T1}, B::COOMatrix{T2}) where {T1, T2}
    # Check that the inner dimensions match
    @assert size(A, 2) == size(B, 1)
    
    # Initialize empty arrays for the result matrix
    rowval = Int[]
    colval = Int[]
    nzval = promote_type(T1, T2)[]  # Promote types to handle mixed precision
    
    # Iterate through all non-zero elements of A
    for (i, j, v) in zip(A.rowval, A.colval, A.nzval)
        # For each non-zero element in A, find matching elements in B
        for (i2, j2, v2) in zip(B.rowval, B.colval, B.nzval)
            # If the column index of A matches the row index of B
            if j == i2
                # This creates a non-zero element in the result matrix
                push!(rowval, i)
                push!(colval, j2)
                push!(nzval, v * v2)  # Multiply the values
            end
        end
    end
    
    # Create and return the result matrix in COO format
    # Note: This implementation doesn't combine duplicate entries
    return COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval)
end
```

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


```julia
using BenchmarkTools

stiffmatrix(n::Int, C) = COOMatrix(n, n, [1:n-1; 1:n; 2:n], [2:n; 1:n; 1:n-1], [C*ones(n-1); -C ; -2C*ones(n-2); -C; C*ones(n-1)])

matrix = stiffmatrix(10000, 1.0)
x = randn(size(matrix, 2))
@btime matrix * x         # 27.292 μs
@btime matrix * matrix    # 800.915 ms
```

== Compressed Sparse Column (CSC) format

The Compressed Sparse Column (CSC) format is one of the most widely used sparse matrix storage formats, especially in scientific computing. It provides efficient column-wise access and matrix operations while minimizing memory usage.

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

The `m`, `n`, `rowval` and `nzval` have the same meaning as those in the COO format. `colptr` is an integer vector of size $n+1$, and `colptr[j]` points to the first nonzero element in the $j$-th column. Hence the $j$-th column of the matrix is stored as a tuple:
- `rowval[colptr[j]:colptr[j+1]-1]`, the row indices of the nonzero elements in the $j$-th column,
- `nzval[colptr[j]:colptr[j+1]-1]`, the nonzero values in the $j$-th column.

This column-oriented structure makes CSC format particularly efficient for column-wise operations and matrix-vector multiplication. Julia's `SparseArrays` standard library uses CSC as its primary sparse matrix format.
A CSC format sparse matrix can be constructed from the COO format with the `SparseArrays.sparse` function:
```julia
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
The output has data type `SparseMatrixCSC{Int64, Int64}`, where `Int64` is the type of the matrix elements. In the following, we implement a CSC matrix from scratch for better understanding:
```julia
struct CSCMatrix{Tv,Ti} <: AbstractMatrix{Tv}
    m::Int              # Number of rows
    n::Int              # Number of columns
    colptr::Vector{Ti}  # Column pointers (length n+1)
    rowval::Vector{Ti}  # Row indices of non-zero elements
    nzval::Vector{Tv}   # Values of non-zero elements
    function CSCMatrix(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        # Validate that colptr has the correct length (n+1)
        @assert length(colptr) == n + 1
        # Validate that rowval and nzval have the same length and match the last colptr entry
        @assert length(rowval) == length(nzval) == colptr[end] - 1
        new{Tv, Ti}(m, n, colptr, rowval, nzval)
    end
end

# Return the dimensions of the matrix
Base.size(A::CSCMatrix) = (A.m, A.n)
# Return a specific dimension of the matrix
Base.size(A::CSCMatrix, i::Int) = getindex((A.m, A.n), i)
# Return the number of non-zero elements in the matrix
nnz(csc::CSCMatrix) = length(csc.nzval)

# Convert a COO matrix to a CSC matrix
function CSCMatrix(coo::COOMatrix{Tv, Ti}) where {Tv, Ti}
    m, n = size(coo)
    # Sort the COO matrix entries by column-major order (column first, then row)
    order = sortperm(1:nnz(coo); by=i->coo.rowval[i] + m * (coo.colval[i]-1))
    # Initialize arrays for CSC format
    colptr, rowval, nzval = similar(coo.rowval, n+1), similar(coo.rowval), similar(coo.nzval)
    k = 0  # Counter for unique entries after accumulation
    ipre, jpre = 0, 0  # Previous row and column indices
    colptr[1] = 1  # First column starts at index 1
    
    # Process each entry in sorted order
    for idx in order
        i, j, v = coo.rowval[idx], coo.colval[idx], coo.nzval[idx]
        # If this entry has the same indices as the previous one, accumulate values
        if i == ipre && j == jpre
            nzval[k] += v
        else
            # New unique entry
            k += 1
            # If we've moved to a new column, update column pointers
            if j != jpre
                # Fill all column pointers from previous column up to current column
                colptr[jpre+1:j+1] .= k
            end
            # Store the row index and value
            rowval[k] = i
            nzval[k] = v
            # Update previous indices
            ipre, jpre = i, j
        end
    end
    
    # Fill remaining column pointers
    colptr[jpre+1:end] .= k + 1
    # Resize arrays to actual number of unique entries
    resize!(rowval, k)
    resize!(nzval, k)
    
    return CSCMatrix(m, n, colptr, rowval, nzval)
end

# Implement indexing for CSC matrix (A[i,j])
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

The row indices and values of nonzero elements in the 3rd column can be obtained by
```julia
cscm = CSCMatrix(matrix)
# Get row indices of nonzeros in column 3
rows3 = cscm.rowval[cscm.colptr[3]:cscm.colptr[4]-1]
# Get values of nonzeros in column 3
val3 = cscm.nzval[cscm.colptr[3]:cscm.colptr[4]-1]
# Alternatively, use the nzrange helper function
cscm.rowval[nzrange(cscm, 3)] # equivalent to the first approach
```

This demonstrates the core advantage of the CSC format: direct access to all nonzeros in a specific column.

```julia
# Matrix-vector multiplication for CSC matrices
# y = A*x where A is in CSC format
function LinearAlgebra.mul!(y::AbstractVector{T}, A::CSCMatrix, x::AbstractVector{T}) where T
    # Initialize result vector to zeros
    fill!(y, zero(T))
    
    # Loop through each column of A
    for j in 1:size(A, 2)
        # For each nonzero element in column j
        for k in nzrange(A, j)
            # Add contribution to result vector: y[i] += A[i,j] * x[j]
            # where i = A.rowval[k] is the row index of the nonzero element
            y[A.rowval[k]] += A.nzval[k] * x[j]
        end
    end
    return y
end

# Matrix-matrix multiplication for CSC matrices
# C = A*B where both A and B are in CSC format
function Base.:*(A::CSCMatrix{T1}, B::CSCMatrix{T2}) where {T1, T2}
    # Determine the result type by promoting types of A and B
    T = promote_type(T1, T2)
    
    # Check that inner dimensions match
    @assert size(A, 2) == size(B, 1)
    
    # Initialize arrays to store result in COO format
    rowval, colval, nzval = Int[], Int[], T[]
    
    # Loop through each column of B
    for j2 in 1:size(B, 2)  # enumerate the columns of B
        # For each nonzero element in column j2 of B
        for k2 in nzrange(B, j2)  # enumerate the rows of B
            # Get the value of B[i2,j2] where i2 = B.rowval[k2]
            v2 = B.nzval[k2]
            i2 = B.rowval[k2]
            
            # For each nonzero element in column i2 of A
            for k1 in nzrange(A, i2)  # enumerate the rows of A
                # Add contribution to result: C[i1,j2] += A[i1,i2] * B[i2,j2]
                # where i1 = A.rowval[k1]
                push!(rowval, A.rowval[k1])
                push!(colval, j2)
                push!(nzval, A.nzval[k1] * v2)
            end
        end
    end
    
    # Convert from COO to CSC format and return
    return CSCMatrix(COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval))
end
```

Let's test the performance of the CSC matrix:
```julia
@btime cscm * x      # 37.042 μs
@btime cscm * cscm    # 3.349 ms
```
While the matrix-vector multiplication has a similar performance as the COO matrix, the matrix-matrix multiplication is much faster than the COO matrix.
This is because the time complexity of multiplying two CSC matrices $A$ and $B$ is $O(op("nnz")(A)op("nnz")(B)\/n)$, while the time complexity of multiplying two COO matrices is $O(op("nnz")(A)op("nnz")(B))$.

The implementation in `SparseArrays` is even more efficient.

```julia
using SparseArrays
sp = sparse(matrix.rowval, matrix.colval, matrix.nzval, matrix.m, matrix.n)
@btime sp * x     # 26.917 μs
@btime sp * sp    # 255.208 μs
```

== Dominant eigenvalue problem

Given a matrix $A in RR^(n times n)$, the dominant eigenvalue problem is to find the largest (or smallest) eigenvalue $lambda_1$ and its corresponding eigenvector $x_1$:

$ min_(x_1) lambda_1 quad "s.t." A x_1 = lambda_1 x_1. $

The power method is a simple iterative algorithm to solve the dominant eigenvalue problem. The algorithm starts with a random vector $v_0$ and repeatedly multiplies it with the matrix $A$.

$ v_k = A^k v_0 $

By representing the initial vector $v_0$ as a linear combination of eigenvectors of $A$, i.e. $v_0 = sum_(i=1)^n c_i x_i$, we have

$ v_k = sum_(i=1)^n lambda_i^k c_i x_i $

where $lambda_1 > lambda_2 >= dots.h >= lambda_n$ are the eigenvalues of $A$ and $x_i$ are the corresponding eigenvectors. The power method converges to the eigenvector corresponding to the largest eigenvalue as $k -> infinity$. The rate of convergence is determined by $|lambda_2/lambda_1|^k$. The Julia code for the power method is as follows.

```julia
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

By inverting the sign, $A -> -A$, we can use the same method to obtain the smallest eigenvalue.

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

#exampleblock[
*Example*: Solve the slowest mode of the spring chain.

We use the `KrylovKit.eigsolve` function. This function accepts a linear map, an initial vector, the number of eigenvalues to compute, and the target eigenvalue type.

```julia
julia> using KrylovKit

julia> eigsolve(sp, randn(size(sp, 1)), 1, :SR)
([-3.999996868663288], ..., ConvergenceInfo: no converged values after 100 iterations and 1218 applications of the linear map;
norms of residuals are given by (6.611549984040687e-5,).
```
Here, we use the `:SR` target eigenvalue type, which means the "smallest" real part of the eigenvalue. The output contains three parts, the eigenvalues, the eigenvectors, and the convergence information. The eigenvalues are not the expected, since the smallest real part can be negative, but we want the one closest to zero.
- _Remark_: Since this matrix is real symmetric, all eigenvalues and all eigenvectors are real. It calls into the Lanczos algorithm.

Let us correct the target eigenvalue type using the `EigSorter`.
```julia
julia> eigsolve(sp, randn(size(sp, 1)), 1, EigSorter(abs; rev=false))
([-3.148596175606358e-6], ..., ConvergenceInfo: no converged values after 100 iterations and 1218 applications of the linear map;
norms of residuals are given by (8.144476258164312e-5,).
)
```
It produces a value $~-3times 10^(-6)$, which is very close to 0 as expected. We do not know if it is a zero eigenvalue, or just being very small, since the residual is still large. Let us improve the precision by setting a larger tolerance and a larger maximum number of iterations:
```julia
julia> eigsolve(sp, randn(size(sp, 1)), 1, EigSorter(abs; rev=false), tol=1e-10, maxiter=5000)
([-4.778184048570348e-8], ..., ConvergenceInfo: no converged values after 5000 iterations and 60018 applications of the linear map;
norms of residuals are given by (1.0713185997288401e-6,).
```
Now the residual is much smaller, and the eigenvalue is also two orders of magnitude smaller.
Hence, we are more confident that it is a zero eigenvalue.
]

== The Krylov subspace method

The Krylov subspace method is the method that `KrylovKit.jl` uses to solve the large scale dominant eigenvalue problem. The input can be any kind of linear map that implements the `mul!` interface, which performs the matrix-vector multiplication. The linear map does not have to be a dense or sparse matrix, but can be any kind of linear operator $A$ satisfying:

$ A (alpha x + beta y) = alpha A(x) + beta A(y). $

The Krylov subspace method, such as the Arnoldi and Lanczos algorithms, have much faster convergence speed comparing with the power method.
The key idea is to generate an orthogonal matrix $Q in CC^(n times k)$ with $k << n$, $Q^dagger Q = I$, such that
$ Q^dagger A Q = B, $
and the largest eigenvalue of $B$ best approximates the largest eigenvalue of $A$.
Since $B$ has size $k times k$, the new eigenvalue problem is much easier to solve. The largest eigenvalue of $B$ upper bounds the largest eigenvalue of $A$:
$ lambda_1(B) <= lambda_1(A), $
where $lambda_1(A)$ denotes the largest eigenvalue of $A$. The equality holds if $Q$ is chosen such that $op("span")(Q)$ contains the dominant eigenvectors of $A$.
Whenever, we have $B y_1 = lambda_1(B) y_1$, we have $y_1^dagger Q^dagger A Q y_1 = lambda_1(B) y_1^dagger y_1 = lambda_1(A)$. Then, either $Q y_1$ is the largest eigenvector of $A$, or $A$ has an eigenvalue $lambda_1(A)$ such that $lambda_1(A) > lambda_1(B)$.

The $Q$ can be generated from the *Krylov subspace* generated from a random initial vector $q_1$:
$ cal(K)(A, q_1, k) = op("span"){q_1, A q_1, A^2 q_1, dots, A^(k-1) q_1}. $ <eq:krylov-subspace>
Unlike the power method, the Krylov subspace method generates an orthogonal matrix $Q$ by orthonormalizing the Krylov vectors, rather than just using the last vector. Hence, it is strictly better than the power method.

== The Lanczos algorithm

The Lanczos algorithm is a special case of the Krylov subspace method that designed for symmetric linear operators. It is an iterative process to generate the subspace spanned by the Krylov vectors in @eq:krylov-subspace. Given a symmetric linear operator $A$ on $RR^n$, the the Lanczos algorithm generates an orthogonal matrix $Q$ such that

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

The Lanczos algorithm is basically a Gram-Schmidt orthogonalization process applied to the Krylov subspace:
1. We start with a normalized vector $q_1$ and compute $A q_1$ (the second vector in the Krylov subspace).
2. To find $alpha_1$, we project $A q_1$ onto $q_1$ by computing the inner product:
   $ alpha_1 = q_1^dagger A q_1 $
3. The remainder $r_1 = A q_1 - alpha_1 q_1$ is orthogonal to $q_1$. We set $beta_1 = ||r_1||_2$ and $q_2 = r_1\/beta_1$ (if $beta_1 != 0$).
4. For subsequent steps, we compute $A q_k$ and make it orthogonal to both $q_k$ and $q_(k-1)$ by subtracting their projections:
   $ r_k = A q_k - alpha_k q_k - beta_(k-1) q_(k-1) $

The key insight is that, for symmetric matrices, $r_k$ is automatically orthogonal to $q_1, q_2, dots, q_(k-2)$ due to the properties of the Krylov subspace. This is why we only need to explicitly orthogonalize against the two most recent vectors.

The iteration terminates when $beta_k = 0$, it means the residual vector is zero, indicating that the Krylov subspace has become invariant under $A$. In this case, we have found an exact invariant subspace, and the algorithm terminates. The resulting tridiagonal matrix takes a block-diagonal form:

$ T( "when " beta_j = 0) = mat(
  alpha_1, beta_1, 0, dots, 0;
  beta_1^*, alpha_2, beta_2, dots, 0;
  0, beta_2^*, alpha_3, dots, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, beta_(j-1)^*, alpha_j
), $

This block structure reflects the fact that the Krylov subspace has split into invariant subspaces of $A$.

In practice, the Gram-Schmidt process is numerically unstable, and the orthogonalization process is often replaced by a more numerically stable process, such as the modified Gram-Schmidt process or iterative reorthogonalization.

=== A Julia implementation

```julia
function lanczos(A, q1::AbstractVector{T}; abstol, maxiter) where T
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage for basis vectors and tridiagonal matrix elements
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    
    # Compute first residual: r₁ = Aq₁ - α₁q₁
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Main Lanczos iteration
    for k = 2:min(length(q1), maxiter)
        # Compute next basis vector: q_k = r_{k-1}/β_{k-1}
        push!(q, rk ./ β[k-1])
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        # This enforces orthogonality to the previous two vectors
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual for the off-diagonal element
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == length(q1)
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end
```

=== Reorthogonalization

In the Lanczos algorithm, numerical errors can accumulate and cause the orthogonality of the basis vectors to deteriorate. Reorthogonalization is a technique to maintain orthogonality among these vectors.

One effective approach uses Householder transformations. Let's understand how this works:

1. Consider linearly independent vectors $r_0, dots, r_(k-1) in CC^n$.

2. For each vector $r_i$, we can construct a Householder matrix $H_i$ that reflects vectors across a hyperplane.

3. When we apply the sequence of Householder matrices $H_0, dots, H_(k-1)$ to the matrix $(r_0|dots|r_(k-1))$, we get:
   $(H_0 dots H_(k-1))^T (r_0|dots|r_(k-1)) = R$
   where $R$ is an upper triangular matrix.

4. If we denote the first $k$ columns of the matrix product $(H_0 dots H_(k-1))$ as $(q_1 | dots | q_k)$, then these vectors $q_1, dots, q_k$ form an orthonormal basis.

This approach is numerically stable and ensures that orthogonality is maintained to machine precision, which is crucial for the convergence and accuracy of the Lanczos algorithm.
```julia
# Lanczos algorithm with explicit reorthogonalization using Householder transformations
function lanczos_reorthogonalize(A, q1::AbstractVector{T}; abstol, maxiter) where T
    n = length(q1)
    
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Store Householder transformations for reorthogonalization
    householders = [householder_matrix(q1)]
    
    # Main Lanczos iteration with reorthogonalization
    for k = 2:min(n, maxiter)
        # Step 1: Apply all previous Householder transformations to residual vector
        # This ensures full orthogonality to all previous vectors
        for j = 1:k-1
            left_mul!(view(rk, j:n), householders[j])
        end
        
        # Create new Householder transformation for the current residual
        push!(householders, householder_matrix(view(rk, k:n)))
        
        # Step 2: Compute the k-th orthonormal vector by applying Householder transformations
        # Start with unit vector e_k and apply all Householder transformations in reverse
        qk = zeros(T, n)
        qk[k] = 1  # qₖ = H₁H₂…Hₖeₖ
        for j = k:-1:1
            left_mul!(view(qk, j:n), householders[j])
        end
        push!(q, qk)
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == n
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end

# Householder transformation matrix representation
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}    # Householder vector
    β::T            # Scaling factor
end

# Apply Householder transformation: B = (I - β*v*v')*B
function left_mul!(B, A::HouseholderMatrix)
    # Compute v'*B
    vB = A.v' * B
    # Apply transformation: B = B - β*v*(v'*B)
    B .-= (A.β .* A.v) * vB
    return B
end

# Create a Householder matrix that transforms v to a multiple of e₁
function householder_matrix(v::AbstractVector{T}) where T
    v = copy(v)
    # Modify first element to ensure numerical stability
    v[1] -= norm(v, 2)
    # Compute scaling factor β = 2/||v||²
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end
```

In the following example, we use the Lanczos algorithm to find the eigenvalues of a graph Laplacian.
```julia
using Graphs

# Create a random 3-regular graph with 1000 vertices
n = 1000
graph = random_regular_graph(n, 3)

# Get the Laplacian matrix of the graph
A = laplacian_matrix(graph)

# Generate a random initial vector
q1 = randn(n)

# Apply our Lanczos implementation
T, Q = lanczos_reorthogonalize(A, q1; abstol=1e-5, maxiter=100)

# Compute eigenvalues of the resulting tridiagonal matrix
eigenvalues = eigen(T).values

# Compare with KrylovKit.jl implementation
using KrylovKit

# Find the two smallest eigenvalues using KrylovKit
# :SR means "smallest real part"
vals, vecs, info = eigsolve(A, q1, 2, :SR)
println("Two smallest eigenvalues: ", vals, ", compared with our implementation: ", eigenvalues[1:2])
# output:
# Two smallest eigenvalues: [3.1110528609336337e-15, 0.17548338667817945], compared with our implementation: [4.440892098500626e-15, 0.17548398110263008]
```

=== Essential aspects of a professional Lanczos implementation
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

```julia
function arnoldi_iteration(A::AbstractMatrix{T}, x0::AbstractVector{T}; maxiter) where T
    # Storage for Hessenberg matrix entries (column by column)
    h = Vector{T}[]
    # Storage for orthonormal basis vectors of the Krylov subspace
    q = [normalize(x0)]
    n = length(x0)
    # Ensure A is a square matrix of appropriate dimensions
    @assert size(A) == (n, n)
    
    # Main Arnoldi iteration loop
    for k = 1:min(maxiter, n)
        # Apply the matrix to the latest basis vector
        u = A * q[k]    # generate next vector
        
        # Initialize the k-th column of the Hessenberg matrix
        hk = zeros(T, k+1)
        
        # Orthogonalize against all previous basis vectors (Gram-Schmidt process)
        for j = 1:k # subtract from new vector its components in all preceding vectors
            hk[j] = q[j]' * u  # Calculate projection coefficient
            u = u - hk[j] * q[j]  # Subtract projection
        end
        
        # Calculate the norm of the remaining vector
        hkk = norm(u)
        hk[k+1] = hkk  # This will be the subdiagonal entry
        push!(h, hk)  # Store this column of coefficients
        
        # Check for convergence or breakdown
        if abs(hkk) < 1e-8 || k >= n # stop if matrix is reducible
            break
        else
            # Normalize the new basis vector and add to our collection
            push!(q, u ./ hkk)
        end
    end

    # Construct the Hessenberg matrix H from the stored coefficients
    kmax = length(h)
    H = zeros(T, kmax, kmax)
    for k = 1:length(h)
        if k == kmax
            # Last column might be shorter if we had early termination
            H[1:k, k] .= h[k][1:k]
        else
            # Standard case: copy the full column including subdiagonal entry
            H[1:k+1, k] .= h[k]
        end
    end
    
    # Return the Hessenberg matrix and the orthonormal basis matrix
    return H, hcat(q...)
end
```

In the following example, we use the Arnoldi algorithm to find the eigenvalues of a random matrix.
```julia
using SparseArrays, LinearAlgebra, KrylovKit

# Create a sparse random matrix with 10% non-zero entries
n = 100
A = sprand(n, n, 0.1)

# Create a random starting vector and normalize it
q1 = randn(n)
q1 = q1 / norm(q1)

# Run our Arnoldi iteration implementation
h, q = arnoldi_iteration(A, q1; maxiter=20)

# Compare eigenvalues from our implementation with KrylovKit
evals_arnoldi = eigen(h).values
# output:
# ComplexF64[..., 1.431963042798233 + 0.6076924727135001im, 4.715884744990036 + 0.0im]

# Compare with KrylovKit.jl implementation
evals_krylovkit, evecs, info = eigsolve(A, q1, 2, :LR)
evals_krylovkit
# output:
# 3-element Vector{ComplexF64}:
#   4.715884731681078 + 0.0im
#  1.5643884820156855 + 0.10220803623279921im
#  1.5643884820156855 - 0.10220803623279921im

# Check the residual for the largest eigenvalue
λ, v = evals_krylovkit[1], evecs[1]
residual = norm(A*v - λ*v) / abs(λ)
println("\nResidual for largest eigenvalue: $residual")
# output:
# Residual for largest eigenvalue: 1.8637188167771364e-15
```
- _Remark_: Sometimes, the `KrylovKit.jl` returns more than the requested number of eigenvalues. Most of the time, the extra eigenvalues are due to the degeneracy of the eigenvalues. They converge simultaneously.

== Graphs

A graph is a pair $G = (V, E)$, where $V$ is a set of vertices and $E$ is a set of edges. In Julia, the package #link("https://github.com/JuliaGraphs/Graphs.jl")[`Graphs.jl`] provides a simple graph data structure. The following code creates a simple graph with 10 vertices.

```julia
using Graphs
g = SimpleGraph(10)  # create an empty graph with 10 vertices
add_vertex!(g)       # add a vertex
add_edge!(g, 3, 11)  # add an edge between vertex 3 and 11
has_edge(g, 3, 11)   # output: true
neighbors(g, 3)      # output: [11]
vertices(g)          # output: OneTo(10)
edges(g)             # output an iterator with element type `SimpleEdge{Int64}`
```

A graph $G = (V, E)$ can be represented by a binary adjacency matrix $A in ZZ_2^(|V| times |V|)$ as
$
A_(i j) = cases(1\, quad (i,j) in E, 0\, quad "otherwise"),
$

#figure(canvas(length:0.8cm, {
  import draw: *
  let vrotate(v, theta) = {
    let (x, y) = v
    return (x * calc.cos(theta) - y * calc.sin(theta), x * calc.sin(theta) + y * calc.cos(theta))
  }

  // petersen graph
  let vertices1 = range(5).map(i=>vrotate((0, 2), i*72/180*calc.pi))
  let vertices2 = range(5).map(i=>vrotate((0, 1), i*72/180*calc.pi))
  let edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (6, 8), (7, 9), (8, 5), (9, 6))
  show-graph((vertices1 + vertices2).map(v=>(v.at(0) + 4, v.at(1)+4)), edges, radius:0.2)
}), caption: "The Petersen graph") <fig:petersen>

For example, the adjacency matrix of the Petersen graph in @fig:petersen is

```julia
using Graphs
graph = smallgraph(:petersen)
adj_matrix = adjacency_matrix(graph)
# output:
# 10×10 SparseArrays.SparseMatrixCSC{Int64, Int64} with 30 stored entries:
#  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  ⋅
#  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
#  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅
#  ⋅  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅
#  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  1
#  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1  ⋅
#  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1
#  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  1
#  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  ⋅
#  ⋅  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅
```

The Laplacian matrix $L$ of a graph $G$ is defined as $L = D - A$, where $D$ is the degree matrix of the graph. The degree matrix is a diagonal matrix, where the diagonal element $D_(i i)$ is the degree of vertex $i$. The Laplacian matrix is symmetric and positive semidefinite.

```julia
lap_matrix = laplacian_matrix(graph)
```

=== Shortest path problem - The tropical matrix multiplication approach
The shortest path problem is to find the shortest path between two vertices in a graph. The tropical matrix multiplication approach@Moore2011 is one of the most efficient ways to solve the shortest path problem.

It can be solved directly with the Min-Plus Tropical matrix multiplication:

$ (A B)_(i k) = min_j (A_(i j) + B_(j k)). $

By powering the adjacency matrix $A$ for $|V|$ times with Min-Plus Tropical algebra, we can get the shortest paths length between any two vertices.
$
  (A^(|V|))_(i j) = min_(k_1, k_2, dots, k_(|V|-1)) (A_(i k_1) + A_(k_1 k_2) + dots + A_(k_(|V|-1) j))
$

The implementation of the tropical matrix multiplication is straightforward.
```julia
using TropicalNumbers, LinearAlgebra
tmat = map(x->iszero(x) ? zero(TropicalMinPlus{Float64}) : TropicalMinPlus(1.0), adjacency_matrix(g))  # TropicalMinPlus zero is Inf.
tmat += Diagonal(fill(TropicalMinPlus(0.0), nv(g)))  # set diagonal to 0
tmat^(nv(graph))
# output:
# 10×10 SparseArrays.SparseMatrixCSC{TropicalMinPlusF64, Int64} with 100 stored entries:
#  0.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ
#  1.0ₛ  0.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ
#  2.0ₛ  1.0ₛ  0.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ
#  2.0ₛ  2.0ₛ  1.0ₛ  0.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ
#  1.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  0.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ
#  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  0.0ₛ  2.0ₛ  1.0ₛ  1.0ₛ  2.0ₛ
#  2.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  0.0ₛ  2.0ₛ  1.0ₛ  1.0ₛ
#  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  0.0ₛ  2.0ₛ  1.0ₛ
#  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  1.0ₛ  1.0ₛ  2.0ₛ  0.0ₛ  2.0ₛ
#  2.0ₛ  2.0ₛ  2.0ₛ  2.0ₛ  1.0ₛ  2.0ₛ  1.0ₛ  1.0ₛ  2.0ₛ  0.0ₛ
```

To get the shortest path length between vertex 1 and other vertices, we simply read the first row of the result. To confirm the result, we can use the built-in function in `Graphs.jl`:

```julia
dijkstra_shortest_paths(g, 2)
# output:
# Graphs.DijkstraState{Int64, Int64}([2, 0, 2, 3, 1, 1, 2, 3, 7, 7], [1, 0, 1, 2, 2, 2, 1, 2, 2, 2], [Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[]], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], Int64[])
```

#bibliography("refs.bib")