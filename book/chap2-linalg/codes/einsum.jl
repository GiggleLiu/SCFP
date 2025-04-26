using OMEinsum

s = fill(1)  # scalar
w, v = [1, 2], [4, 5];  # vectors
A, B = [1 2; 3 4], [5 6; 7 8]; # matrices
T1, T2 = reshape(1:8, 2, 2, 2), reshape(9:16, 2, 2, 2); # 3D tensor

# Single tensor operations
ein"i->"(w)  # sum of the elements of a vector.
ein"ij->i"(A)  # sum of the rows of a matrix.
ein"ii->"(A)  # sum of the diagonal elements of a matrix, i.e., the trace.
ein"ij->"(A)  # sum of the elements of a matrix.
ein"i->ii"(w)  # create a diagonal matrix.
ein",->"(s, s)  # scalar multiplication.

# Two tensor operations
ein"ij, jk -> ik"(A, B)  # matrix multiplication.
ein"ijb,jkb->ikb"(T1, T2)  # batch matrix multiplication.
ein"ij,ij->ij"(A, B)  # element-wise multiplication.
ein"ij,ij->"(A, B)  # sum of the element-wise multiplication.
ein"ij,->ij"(A, s)  # element-wise multiplication by a scalar.

# More than two tensor operations
optein"ai,aj,ak->ijk"(A, A, B)  # star contraction.
optein"ia,ajb,bkc,cld,dm->ijklm"(A, T1, T2, T1, A)  # tensor train contraction.

