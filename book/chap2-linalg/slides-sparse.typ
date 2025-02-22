Consider the stiffness matrix of a spring chain, which is tridiagonal and has only $3n-2$ nonzero elements.

$
mat(
  -C, C, 0, dots.h, 0;
  C, -2C, C, dots.h, 0;
  0, C, -2C, dots.h, 0;
  dots.v, dots.v, dots.v, dots.down, dots.v;
  0, 0, 0, C, -C
)
$

Storing such a matrix in a dense format requires $n^2$ elements, which is very memory inefficient since it has only $3n-2$ nonzero elements.
