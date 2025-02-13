#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "../book.typ": book-page

#set math.equation(numbering: "(1)")

#show: book-page.with(title: "Automatic differentiation")
#show: thmrules

#import "@preview/ouset:0.2.0": ouset

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em, bottom: 1em), base: none, stroke: black)
#let theorem = thmbox("theorem", "Theorem", base: none, stroke: black)
#let proof = thmproof("proof", "Proof")

= Automatic Differentiation

Automatic differentiation@Griewank2008 is a technique to compute the derivative of a function automatically. It is a powerful tool for scientific computing, machine learning, and optimization. The automatic differentiation can be classified into two types: forward mode and backward mode. The forward mode AD computes the derivative of a function with respect to many inputs, while the backward mode AD computes the derivative of a function with respect to many outputs. The forward mode AD is efficient when the number of inputs is small, while the backward mode AD is efficient when the number of outputs is small.

== A brief history of autodiff

- 1964 (*forward mode AD*) ~ Robert Edwin Wengert, A simple automatic derivative evaluation program.
- 1970 (*backward mode AD*) ~ Seppo Linnainmaa, Taylor expansion of the accumulated rounding error.
- 1986 (*AD for machine learning*) ~ Rumelhart, D. E., Hinton, G. E., and Williams, R. J., Learning representations by back-propagating errors.
- 1992 (*optimal checkpointing*) ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.
- 2000s ~ The boom of tensor based AD frameworks for machine learning.
- 2020 (*AD on LLVM*) ~ Moses, William and Churavy, Valentin, Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients.

== Differentiating the Bessel function

$ J_nu(z) = sum_(n=0)^infinity (z/2)^nu/(Gamma(k+1)Gamma(k+nu+1)) (-z^2/4)^n $

A poor man's implementation of this Bessel function is as follows:

```julia
function poor_besselj(ν, z::T; atol=eps(T)) where T
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k+ν) * (z/2)^2
        out += s
    end
    out
end
```

Let us plot the Bessel function:

#let plot-bessel = {
  // Note: This is a placeholder for the actual plotting code
  // since Typst doesn't directly support Julia plotting
  // Consider using an external plotting tool and importing the image
}

The derivative of the Bessel function is:
$ (d J_nu(z))/(d z) = (J_(nu-1)(z) - J_(nu+1)(z))/2 $

== Finite difference
The finite difference is a numerical method to approximate the derivative of a function. The finite difference can be classified into three types: forward, backward, and central.

For example, the first order forward difference is:
$ (partial f)/(partial x) approx (f(x+Delta) - f(x))/(Delta) $

The first order backward difference is:
$ (partial f)/(partial x) approx (f(x) - f(x-Delta))/(Delta) $

The first order central difference is:
$ (partial f)/(partial x) approx (f(x+Delta) - f(x-Delta))/(2Delta) $

Among these three methods, the central difference is the most accurate. It has an error of $O(Delta^2)$, while the forward and backward differences have an error of $O(Delta)$.

Higher order finite differences can be found in the #link("https://en.wikipedia.org/wiki/Finite_difference_coefficient")[wiki page].

#box(
  fill: rgb("e5e5e5"),
  inset: 1em,
  radius: 4pt,
)[
  *Example: central finite difference to the 4th order*

  The coefficients of the central finite difference to the 4th order are:

  #table(
    columns: 5,
    [-2], [-1], [0], [1], [2],
    [1/12], [-2/3], [0], [2/3], [-1/12]
  )

  The induced formula is:
  $ (partial f)/(partial x) approx (f(x-2Delta) - 8f(x-Delta) + 8f(x+Delta) - f(x+2Delta))/(12Delta) $

  In the following, we will derive this formula using the Taylor expansion.
  $ vec(
    f(x-2Delta), f(x-Delta), f(x), f(x+Delta), f(x+2Delta)
  ) approx mat(
    1, (-2)^1, (-2)^2, (-2)^3, (-2)^4;
    1, (-1)^1, (-1)^2, (-1)^3, (-1)^4;
    1, 0, 0, 0, 0;
    1, (1)^1, (1)^2, (1)^3, (1)^4;
    1, (2)^1, (2)^2, (2)^3, (2)^4
  ) vec(
    f(x), f'(x)Delta, f''(x)Delta^2/2, f'''(x)Delta^3/6, f''''(x)Delta^4/24
  ) $
]

== Forward mode automatic differentiation

Forward mode AD attaches a infitesimal number $epsilon$ to a variable, when applying a function $f$, it does the following transformation
$ f(x+g epsilon) = f(x) + f'(x) g epsilon + cal(O)(epsilon^2) $

The higher order infinitesimal is ignored. 

*In the program*, we can define a *dual number* with two fields, just like a complex number
$ f((x, g)) = (f(x), f'(x)*g) $

```julia
using ForwardDiff
res = sin(ForwardDiff.Dual(π/4, 2.0))
res === ForwardDiff.Dual(sin(π/4), cos(π/4)*2.0)
```


We can apply this transformation consecutively, it reflects the chain rule.
$ (partial vec(y)_(i+1))/(partial x) = ((partial vec(y)_(i+1))/(partial vec(y)_i)) (partial vec(y)_i)/(partial x) $



*Example:* Computing two gradients $(partial z sin x)/(partial x)$ and $(partial sin^2 x)/(partial x)$ at one sweep

```julia-repl
julia> autodiff(Forward, poor_besselj, 2, Duplicated(0.5, 1.0))[1]
0.11985236384014333

julia> @benchmark autodiff(Forward, poor_besselj, 2, Duplicated(x, 1.0))[1] setup=(x=0.5)
BenchmarkTools.Trial: 10000 samples with 996 evaluations.
 Range (min … max):  22.256 ns … 66.349 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     23.050 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   23.290 ns ±  1.986 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▅▅  █▆▂▂▄▂▂▂▁                                               ▁
  ██▅▇██████████▅▅▄▅▅▅▅▃▄▄▄▅▆▅▅▃▄▃▄▅▄▅▄▆▅▅▄▃▂▄▃▃▂▃▄▃▄▄▃▂▂▃▂▃▃ █
  22.3 ns      Histogram: log(frequency) by time      31.8 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

The computing time grows **linearly** as the number of variables that we want to differentiate. But does not grow significantly with the number of outputs.

== Reverse mode automatic differentiation

On the other side, the back-propagation can differentiate **many inputs** with respect to a **single output** efficiently

```math
\begin{align*}
    \frac{\partial \mathcal{L}}{\partial \vec y_i} = \frac{\partial \mathcal{L}}{\partial \vec y_{i+1}}&\boxed{\frac{\partial \vec y_{i+1}}{\partial \vec y_i}}\\
&\text{local jacobian?}
\end{align*}
```

```julia-repl
julia> autodiff(Enzyme.Reverse, poor_besselj, 2, Enzyme.Active(0.5))[1]
(nothing, 0.11985236384014332)

julia> @benchmark autodiff(Enzyme.Reverse, poor_besselj, 2, Enzyme.Active(x))[1] setup=(x=0.5)
BenchmarkTools.Trial: 10000 samples with 685 evaluations.
 Range (min … max):  182.482 ns … 503.771 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     208.880 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   210.059 ns ±  17.016 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▃                 ▁▁█▃                                        
  █▂▁▂▃▁▁▁▁▂▂▁▁▁▁▁▁▃████▄▃▄▄▃▂▂▃▂▃▃▃▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  182 ns           Histogram: frequency by time          260 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

Computing local Jacobian directly can be expensive. In practice, we can use the back-propagation rules to update the adjoint of the variables directly. It requires the forward pass storing the intermediate variables.

== Rule based AD and source code transformation

#table(
  columns: 3,
  [*Type*], [*Rule based*], [*Source code transformation*],
  [Description], 
  [Defining backward rules manually for functions on tensors], 
  [Defining backward rules on a limited set of basic scalar operations, and generate gradient code using source code transformation],
  [Pros and cons],
  [
    - #text(green)[Good tensor performance]
    - #text(green)[Mature machine learning ecosystem]
  ],
  [
    - #text(green)[Reasonable scalar performance]
    - #text(red)[Automatically generated backward rules]
  ]
)

== Deriving the backward rules for linear algebra

Many backward rules could be found in the notes@Giles2008@Seeger2017. Here we list some of the latest improvements.

=== Matrix multiplication
Let $cal(T)$ be a stack, and $x arrow.r cal(T)$ and $x arrow.l cal(T)$ be the operation of pushing and poping an element from this stack.
Given $A in R^(l times m)$ and $B in R^(m times n)$, the forward pass computation of matrix multiplication is
$ 
cases(
  C = A B,
  A arrow.r cal(T),
  B arrow.r cal(T),
  dots
)
$

Let the adjoint of $x$ be $overline(x) = (partial cal(L))/(partial x)$, where $cal(L)$ is a real loss as the final output.
The backward pass computes
$ 
cases(
  dots,
  B arrow.l cal(T),
  overline(A) = overline(C)B,
  A arrow.l cal(T),
  overline(B) = A overline(C)
)
$

The rules to compute $overline(A)$ and $overline(B)$ are called the backward rules for matrix multiplication. They are crucial for rule based automatic differentiation.

Let us introduce a small perturbation $delta A$ on $A$ and $delta B$ on $B$,

$ delta C = delta A B + A delta B $

$ delta cal(L) = "tr"(delta C^T overline(C)) = 
"tr"(delta A^T overline(A)) + "tr"(delta B^T overline(B)) $

It is easy to see
$ delta L = "tr"((delta A B)^T overline(C)) + "tr"((A delta B)^T overline(C)) = 
"tr"(delta A^T overline(A)) + "tr"(delta B^T overline(B)) $

We have the backward rules for matrix multiplication as
$ 
cases(
  overline(A) = overline(C)B^T,
  overline(B) = A^T overline(C)
)
$

=== Symmetric Eigen decomposition

Given a symmetric matrix $A$, the eigen decomposition is

$ A = U E U^dagger $

Where $U$ is the eigenvector matrix, and $E$ is the eigenvalue matrix. The backward rules for the symmetric eigen decomposition are@Seeger2017

$ overline(A) = U[overline(E) + 1/2(overline(U)^dagger U circle F + h.c.)]U^dagger $

Where $F_(i j)=(E_j- E_i)^(-1)$.

If $E$ is continuous, we define the density $rho(E) = sum_(k) delta(E-E_k)=-1/pi integral_k Im[G^r(E, k)]$ (check sign!). Where $G^r(E, k) = 1/(E-E_k+i delta)$.

We have
$ overline(A) = U[overline(E) + 1/2(overline(U)^dagger U circle Re[G(E_i, E_j)] + h.c.)]U^dagger $

=== Singular Value Decomposition (SVD)

*References*:

Complex valued SVD is defined as $A = U S V^dagger$. For simplicity, we consider a *full rank square matrix* $A$.
Differentiating the SVD@Wan2019@Francuz2023, we have

$ d A = d U S V^dagger + U d S V^dagger + U S d V^dagger $

$ U^dagger d A V = U^dagger d U S + d S + S d V^dagger V $

Defining matrices $d C=U^dagger d U$ and $d D = d V^dagger V$ and $d P = U^dagger d A V$, then we have

$ cases(
  d C^dagger + d C = 0,
  d D^dagger + d D = 0
) $

We have

$ d P = d C S + d S + S d D $

where $d C S$ and $S d D$ has zero real part in diagonal elements. So that $d S = Re["diag"(d P)]$. 

$ 
  d cal(L) &= "tr"(overline(A)^T d A + overline(A^*)^T d A^*) \
  &= "tr"(overline(A)^T d A + d A^dagger overline(A)^*)
$

Easy to show $overline(A)_s = U^* overline(S) V^T$. Notice here, $overline(A)$ is the *derivative* rather than *gradient*, they are different by a conjugate, this is why we have transpose rather than conjugate here.

Using the relations $d C^dagger + d C = 0$ and $d D^dagger + d D = 0$ 

$ cases(
  d P S + S d P^dagger = d C S^2 - S^2 d C,
  S d P + d P^dagger S = S^2 d D - d D S^2
) $

$ cases(
  d C = F circle (d P S + S d P^dagger),
  d D = -F circle (S d P + d P^dagger S)
) $

where $F_(i j) = 1/(s_j^2 - s_i^2)$, easy to verify $F^T = -F$. Notice here, the relation between the imaginary diagonal parts is lost

$ Im[I circle d P] = Im[I circle (d C + d D)] $

Let's first focus on the off-diagonal contributions from $d U$

$ 
  "tr" overline(U)^T d U &= "tr" overline(U)^T U d C + overline(U)^T (I - UU^dagger) d A V S^(-1) \
  &= "tr" overline(U)^T U (F circle (d P S + S d P^dagger)) \
  &= "tr"(d P S + S d P^dagger)(-F circle (overline(U)^T U)) \
  &= "tr"(d P S + S d P^dagger)J^T
)
$

Here, we defined $J = F circle (U^T overline(U))$.

$ 
  d cal(L) &= "tr"(d P S + S d P^dagger)(J + J^dagger)^T \
  &= "tr" d P S(J + J^dagger)^T + h.c. \
  &= "tr" U^dagger d A V S(J + J^dagger)^T + h.c.
$

By comparing with $d cal(L) = "tr"(overline(A)^T d A + h.c.)$, we have

$ overline(A)_U^("real") = (V S(J + J^dagger)^T U^dagger)^T = U^*(J + J^dagger)S V^T $

=== QR decomposition

Let $A$ be a full rank matrix, the QR decomposition is defined as
$ A = Q R $
with $Q^dagger Q = bb(I)$, so that $d Q^dagger Q + Q^dagger d Q = 0$. $R$ is a complex upper triangular matrix, with diagonal part real.

The backward rules for QR decomposition are derived in multiple references, including @Hubig2019 and @Liao2019. To derive the backward rules, we first consider differentiating the QR decomposition
$ d A = d Q R + Q d R $

$ d Q = d A R^(-1) - Q d R R^(-1) $

$ cases(
  Q^dagger d Q = d C - d R R^(-1),
  d Q^dagger Q = d C^dagger - R^(-dagger)d R^dagger
) $

where $d C = Q^dagger d A R^(-1)$.

Then

$ d C + d C^dagger = d R R^(-1) + (d R R^(-1))^dagger $

Notice $d R$ is upper triangular and its diag is lower triangular, this restriction gives

$ U circle (d C + d C^dagger) = d R R^(-1) $

where $U$ is a mask operator that its element value is $1$ for upper triangular part, $0.5$ for diagonal part and $0$ for lower triangular part. One should also notice here both $R$ and $d R$ has real diagonal parts, as well as the product $d R R^(-1)$.

Now let's wrap up using the Zygote convention of gradient

$ 
  d cal(L) &= "tr"[overline(cal(Q))^dagger d Q + overline(cal(R))^dagger d R + h.c.] \
  &= "tr"[overline(cal(Q))^dagger d A R^(-1) - overline(cal(Q))^dagger Q d R R^(-1) + overline(cal(R))^dagger d R + h.c.] \
  &= "tr"[R^(-1) overline(cal(Q))^dagger d A + R^(-1) M d R + h.c.]
)
$

here, $M = R overline(cal(R))^dagger - overline(cal(Q))^dagger Q$. Plug in $d R$ we have

$ 
  d cal(L) &= "tr"[R^(-1) overline(cal(Q))^dagger d A + M[U compose (d C + d C^dagger)] + h.c.] \
  &= "tr"[R^(-1) overline(cal(Q))^dagger d A + (M compose L)(d C + d C^dagger) + h.c.] \
  &= "tr"[(R^(-1) overline(cal(Q))^dagger d A + h.c.) + (M compose L)(d C + d C^dagger) + (M compose L)^dagger (d C + d C^dagger)] \
  &= "tr"[R^(-1) overline(cal(Q))^dagger d A + (M compose L + h.c.)d C + h.c.]
)
$

== Obtaining Hessian

The second order gradient, Hessian, is also recognized as the Jacobian of the gradient. In practice, we can compute the Hessian by differentiating the gradient function with forward mode AD, which is also known as the forward-over-reverse mode AD.

== Optimal checkpointing
The main drawback of the reverse mode AD is the memory usage. The memory usage of the reverse mode AD is proportional to the number of intermediate variables, which scales linearly with the number of operations. The optimal checkpointing@Griewank2008 is a technique to reduce the memory usage of the reverse mode AD. It is a trade-off between the memory and the computational cost. The optimal checkpointing is a step towards solving the memory wall problem

Given the binomial function $eta(tau, delta) = ((tau + delta)!)/(tau!delta!)$, show that the following statement is true.
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k) $

#bibliography("refs.bib")
