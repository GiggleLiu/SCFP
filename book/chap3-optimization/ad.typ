#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:1.0.3"
#import algorithmic: algorithm
//#import "../book.typ": book-page
#import "images/treeverse.typ": visualize-treeverse

#set math.equation(numbering: "(1)")

//#show: book-page.with(title: "Automatic differentiation")
#show: thmrules

#import "@preview/ouset:0.2.0": ouset

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em, bottom: 1em), base: none, stroke: black)
#let theorem = thmbox("theorem", "Theorem", base: none, stroke: black)
#let proof = thmproof("proof", "Proof")

#align(center, [= Automatic differentiation\
_Jin-Guo Liu_])

== Finite difference
The finite difference is a numerical method to approximate the derivative of a function.
For a scalar function $f: RR arrow RR$, the second order (central) finite difference is given by:
$ (partial f)/(partial x) = (f(x+Delta) - f(x-Delta))/(2Delta) + O(Delta^2), $
where $Delta$ is the discretization step. The estimated gradient has an error of order $O(Delta^2)$.
This definition can be generalized to $2l$-th order with the following form:
$ (partial f)/(partial x) = (sum_(k=-l)^l c_k f(x+k Delta))\/Delta + O(Delta^(2l)), $
where $c_k$ are the coefficients details in @fig:finite-difference-coefficients.

#figure(
  table(
    columns: 9,
    align: center,
    [*Derivative*], [*Accuracy*], [$-3$], [$-2$], [$-1$], [$0$], [$1$], [$2$], [$3$],
    
    table.cell(rowspan: 3)[1], [2], [], [], [$-1/2$], [$0$], [$1/2$], [], [],
    [4], [], [$1/12$], [$-2/3$], [$0$], [$2/3$], [$-1/12$], [],
    [6], [$-1/60$], [$3/20$], [$-3/4$], [$0$], [$3/4$], [$-3/20$], [$1/60$],
    //[8], [$1/280$], [$-4/105$], [$1/5$], [$-4/5$], [$0$], [$4/5$], [$-1/5$], [$4/105$], [$-1/280$],
    
    table.cell(rowspan: 3)[2], [2], [], [], [$1$], [$-2$], [$1$], [], [],
    [4], [], [$-1/12$], [$4/3$], [$-5/2$], [$4/3$], [$-1/12$], [],
    [6], [$1/90$], [$-3/20$], [$3/2$], [$-49/18$], [$3/2$], [$-3/20$], [$1/90$],
    //[8], [$-1/560$], [$8/315$], [$-1/5$], [$8/5$], [$-205/72$], [$8/5$], [$-1/5$], [$8/315$], [$-1/560$],
  ),
  caption: [Finite difference coefficients for first and second derivatives at various accuracy orders. Each row shows the coefficients to be used in the finite difference formula.]
) <fig:finite-difference-coefficients>


#block[
  *Example: deriving the $4$-th order central finite difference*

The coefficients of the central finite difference to the 4th order are:
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

Let us denote the matrix on the right-hand side as $A$. Then we want to find the coefficients $bold(alpha) = (
alpha_(-2), alpha_(-1), alpha_0, alpha_1, alpha_2)$ such that
$
&a_(-2) f(x-2Delta) + a_(-1) f(x-Delta) + a_0 f(x) + a_1 f(x+Delta) + a_2 f(x+2Delta)\
&= f'(x)Delta + O(Delta^5).
$
i.e. all $Delta$ terms of order higher than 1 lower than 5 are canceled.
This can be computed by solving the linear system.
$ A bold(alpha) = (0, 1, 0, 0, 0)^T, $


The following code demonstrates the central finite difference to the 4th order.

```julia
b = [0.0, 1, 0, 0, 0]
A = [i^j for i=-2:2, j=0:4]
A' \ b  # the central_fdm(5, 1) coefficients
```

In Julia, the package `FiniteDifferences.jl` provides a function `central_fdm(5, 1)` to compute the central finite difference to an arbitrary order.
```julia
using FiniteDifferences

central_fdm(5, 1)(x->poor_besselj(2, x), 0.5)
```
]

== Forward mode autodiff
The automatic differentiation (AD) is a technique to compute the derivative of a function represented by a computational process.
It can be classified into two categories: forward mode and reverse mode@Li2017 @Griewank2008.
_Forward mode AD_ presumes the scalar input.
Given a program with scalar input $t$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _derivatives_ as $dot(bold(y)_i) = (partial bold(y)_i)/(partial t)$.
The _forward rule_ defines the transition between $bold(y)_i$ and $bold(y)_(i+1)$
$
dot(bold(y))_(i+1) = (diff bold(y)_(i+1))/(diff bold(y)_i) dot(bold(y))_i.
$
// In the program, we can define a *dual number* with two fields, just like a complex number.
In an automatic differentiation engine, the Jacobian matrix $(diff bold(y)_(i+1))/(diff bold(y)_i)$ is almost never computed explicitly in memory as it can be costly.
Instead, the forward mode automatic differentiation can be implemented by overloading the function $f_i$ as
$ f_i^("forward"): (bold(y)_i, dot(bold(y))_i) arrow.bar (bold(y)_(i+1), (diff bold(y)_(i+1))/(diff bold(y)_i) dot(bold(y))_i), $
which updates both the value and the derivative of the intermediate variables.

In Julia, the package `ForwardDiff.jl`@Revels2016 provides a function `gradient` to compute the gradient of a simple function $f(theta, r) = r cos theta sin theta$.
```julia
using ForwardDiff

theta, r = π/4, 2.0
x = [theta, r]
f(x) = x[2] * cos(x[1]) * sin(x[1])
ForwardDiff.gradient(f, x) # [0.0, 0.5]
```

== Reverse mode autodiff
When we have multiple inputs, the forward mode AD have to repeatedly evaluate the derivatives for each input, which is computationally expensive.
//Let us consider a computational process that computes the value of a function $bold(y) = f(bold(x))$.
To circumvent this issue, the _reverse mode AD_ is proposed, which presumes a scalar output $cal(L)$, or the loss function.
Given a program with scalar output $cal(L)$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _adjoints_ as $overline(bold(y))_i = (partial cal(L))/(partial bold(y)_i)$.
The _backward rule_ defines the transition between $overline(bold(y))_(i+1)$ and $overline(bold(y))_i$
$
overline(bold(y))_i = overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i).
$
Again, in the program, there is no need to compute the Jacobian matrix explicitly in memory.
We define the backward function $overline(f)_i$ as
$ overline(f)_i: ("TAPE", overline(bold(y))_(i+1)) arrow.bar ("TAPE", overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i)), $
where "TAPE" is a cache for storing the intermediate variables that required for implementing the backward rule.
Due to the "TAPE", the reverse mode AD is much harder to implement than the forward mode AD.
The forward mode AD has a natural order of visiting the intermediate variables, which can be supported by running the program forwardly.
While the reverse mode AD has to visit the intermediate variables in the reversed order, we have to run the program forwardly and store the intermediate variables in a stack called "TAPE".
Then in the backward pass, we pop the intermediate variables from the "TAPE" and compute the adjoint of the variables.

As shown in @fig:computational_graph, the computational process can be represented as a directed acyclic graph
(DAG) where nodes are operations and edges are data dependencies.
The forward pass computes the value of the function and stores the intermediate variables in the "TAPE".
The backward pass pops the intermediate variables from the "TAPE" and computes the adjoint of the variables.

#figure((
  canvas({
    import draw: *
    let s(x) = text(8pt, x)
    for (x, y, txt, nm, st) in ((-0.2, 0.5, s[$id$], "t", black), (1, 0, s[$cos$], "cos(t)", black), (1, 1, s[$sin$], "sin(t)", black), (2.5, 0, [$*$], "*", black)) {
      circle((x, y), radius: 0.3, name: nm, stroke: st)
      content((x, y), txt)
    }
    line((rel: (-1, 0), to: "t"), "t", name: "l0")
    line("t", "cos(t)", name: "l1")
    line("t", "sin(t)", name: "l2")
    line("cos(t)", "*", name: "l3")
    line("sin(t)", "*", name: "l4")
    line((rel: (-1, -1), to: "*"), "*", name: "l5")
    line("*", (rel: (1, 0), to: "*"), name: "l6")
    mark("l0.start", "l0.mid", end: "straight")
    mark("l1.start", "l1.mid", end: "straight")
    mark("l2.start", "l2.mid", end: "straight")
    mark("l3.start", "l3.mid", end: "straight")
    mark("l4.start", "l4.mid", end: "straight")
    mark("l5.start", "l5.mid", end: "straight")
    mark("l6.start", "l6.mid", end: "straight")
    content((rel: (0, 0.2), to: "l0.mid"), s[$theta$])
    content((rel: (0, -0.2), to: "l1.mid"), s[$theta$])
    content((rel: (0, 0.2), to: "l2.mid"), s[$theta$])
    content((rel: (0, -0.2), to: "l3.mid"), s[$cos theta$])
    content((rel: (0.2, 0.2), to: "l4.mid"), s[$sin theta$])
    content((rel: (-0.2, -0.2), to: "l6.end"), s[$y$])
    content((rel: (0.1, -0.1), to: "l5.mid"), s[$r$])

    content((1, -1.5), [Forward Pass])

    set-origin((6, 0))
    for (x, y, txt, nm, st) in ((-0.2, 0.5, s[$id$], "t", black), (1, 0, s[$cos$], "cos(t)", black), (1, 1, s[$sin$], "sin(t)", black), (2.5, 0, [$*$], "*", black)) {
      circle((x, y), radius: 0.3, name: nm, stroke: st)
      content((x, y), txt)
    }
    line((rel: (-1, 0), to: "t"), "t", name: "l0")
    line("t", "cos(t)", name: "l1")
    line("t", "sin(t)", name: "l2")
    line("cos(t)", "*", name: "l3")
    line("sin(t)", "*", name: "l4")
    line((rel: (-1, -1), to: "*"), "*", name: "l5")
    line("*", (rel: (1, 0), to: "*"), name: "l6")
    mark("l0.end", "l0.mid", end: "straight")
    mark("l1.end", "l1.mid", end: "straight")
    mark("l2.end", "l2.mid", end: "straight")
    mark("l3.end", "l3.mid", end: "straight")
    mark("l4.end", "l4.mid", end: "straight")
    mark("l5.end", "l5.mid", end: "straight")
    mark("l6.end", "l6.mid", end: "straight")
    content((rel: (-0.7, 0.2), to: "l0.mid"), s[$r (sin^2 theta + cos^2 theta)$])
    content((rel: (-0.3, -0.2), to: "l1.mid"), s[$r sin^2 theta$])
    content((rel: (-0.3, 0.2), to: "l2.mid"), s[$r cos^2 theta$])
    content((rel: (0, -0.2), to: "l3.mid"), s[$r sin theta$])
    content((rel: (0.3, 0.2), to: "l4.mid"), s[$r cos theta$])
    content((rel: (-0.2, -0.2), to: "l6.end"), s[$1$])
    content((rel: (0.6, -0.1), to: "l5.mid"), s[$sin theta cos theta$])

    content((1, -1.5), [Backward Pass])
  })
), caption: [The computational graph for calculating $y = r cos theta sin theta$. Nodes are operations and edges are variables.
The node "$id$" is the copy operation.]) <fig:computational_graph>

In Julia, the package `Enzyme.jl`@Moses2021 provides a state of the art source code transformation based AD engine.
```julia
using Enzyme

gval = zero(x)
_, fval = Enzyme.autodiff(ReverseWithPrimal, f, Active, Duplicated(x, gval))
fval, gval  # 1.0, [0.0, 0.5]
```
The first parameter `ReverseWithPrimal` means return both the gradient and the (primal) function value.
If a scalar variable carries gradient, it should be declared as `Active`, if not, it should be declared as `Const`.
For mutable objects like arrays with gradients, it should be declared as `Duplicated`, with an extra field to store gradients.
Here, the output is a scalar with gradient, hence the thrid field is `Active`, the input is a vector with gradient, hence the last field is a `Duplicated` vector with a mutable gradient field. After calling `Enzyme.autodiff`, the gradient field is changed inplace.


== Obtaining Hessian

The second order gradient, or Hessian, can be computed by taking the Jacobian of the gradient.
Note that the program to compute the gradient of a function is also a differentiable program.
Consider a multivariate function $f: bb(R)^n arrow.r bb(R)$, the gradient function $nabla f: bb(R)^n arrow.r bb(R)^n$ is also a differentiable function.
After computing the gradient with the reverse mode AD, we can use the forward mode AD to compute the Hessian.
The reason why we can use the forward mode AD to compute the Hessian is that the gradient function $nabla f$ has equal number of input and output dimensions.
The forward mode AD is more memory efficient than the reverse mode AD in this case.

#figure(canvas({
  import draw: *
  let s(x) = text(8pt, x)
  for (x, y, txt, nm, st) in ((1.5, 0, s[$f$], "f", black), (3, 0, s[$overline(f)$], "g", black), (4.5, -1, [$*$], "*", black)) {
    circle((x, y), radius: 0.3, name: nm, stroke: st)
    content((x, y), txt)
  }
  line((rel: (-1, 0), to: "f"), "f", name: "l1")
  line("f", "g", name: "l2")
  line("g", "*", name: "l3")
  line((rel: (-1.5, -0.7), to: "*"), "*", name: "l4")
  line("*", (rel: (1, 0)), name: "l5")
  mark("l1.start", "l1.mid", end: "straight")
  mark("l2.start", "l2.mid", end: "straight")
  mark("l3.start", "l3.mid", end: "straight")
  mark("l4.start", "l4.mid", end: "straight")
  mark("l5.start", "l5.mid", end: "straight")

  content((rel: (-0.5, 0.2), to: "l1.mid"), s[$x$])
  content((rel: (0, 0.2), to: "l2.mid"), s[$f(x)$])
  content((rel: (0.2, 0.2), to: "l3.mid"), s[$(diff f)/(diff x)$])
  content((rel: (0.2, -0.1), to: "l4.mid"), s[$v$])
  content((rel: (0.3, 0.0), to: "l5.end"), s[$(diff f)/(diff x)v$])
}), caption: [The computational graph for obtaining the Jacobian vector product.]) <fig:jacobian-vector>

In many algorithms, such as the stochastic reconfiguration @Sorella1998 (or the Dirac-Frenkel variational principle @Raab2000) and the Newton's method, we are only interested in Hessian vector product instead of obtaining the whole matrix. The former can be much cheaper to compute.
The Hessian vector product can be obtained by back-propagating over the Jacobian vector product computational graph as shown in @fig:jacobian-vector, which is:
$
  (diff^2 f)/(diff x^2)v = (diff ((diff f)/(diff x)v))/(diff x)
$ <eq:hvp>
Comparing with obtaining the full Hessian matrix, the computational overhead of @eq:hvp is constant instead of linear to the number of parameters.

In Enzyme, the Hessian vector product can be computed by:

```julia
v = [1.0, 0]
hvp(f, x, v)  # [-4, 0]
```

The full Hessian matrix computed by applying the Hessian vector product to each basis vector:
```julia
using LinearAlgebra
identity_mat = Matrix{Float64}(I, length(x), length(x))
hcat([hvp(f, x, identity_mat[:, i]) for i in 1:length(x)]...)  # [-4 0; 0 0]
```

== Complex valued automatic differentiation <complex-valued-automatic-differentiation>
Complex valued AD considers the problem that a function takes complex variables as inputs, while the loss is still real valued.
Since such function cannot be holomorphic, or complex differentiable, the adjoint of a such a function is defined by treating the real and imaginary parts of the input as independent variables.
Let $z = x + i y$ be a complex variable, and $cal(L)$ be a real loss function.
The adjoint of $z$ is defined as
$
  overline(z) = overline(x) + i overline(y).
$
If we change $z$ by a small amount $delta z = delta x + i delta y$, the loss function $cal(L)$ will change by
$ delta cal(L) = (overline(z)^* delta z + h.c.)\/2 = overline(x) delta x + overline(y) delta y. $


`Enzyme` can be used to compute the gradient of a complex valued function.
```julia
fc(z) = abs2(z)
# a fixed input to use for testing
z = 3.1 + 2.7im
Enzyme.autodiff(Reverse, fc, Active, Active(z))[1][1]  # 6.2 + 5.4im
```
In this example, the input is a complex number $z$, and the output is a real number $|z|^2$. The gradient is $2 z$.

== Rule based AD

The Julia rule based AD ecosystem is composed of two main parts, the rule sets and the engines.
The rule sets define the forward and backward rules for the basic operations,
while the AD engines makes use of the rules to compute the gradient of a general program.
The most popular rule set is `ChainRules`, which is built on top of `ChainRulesCore`.
Users can easily define their own rules by overloading the `ChainRulesCore.rrule` function.
`ChainRulesCore` provides a unified interface for the rules, and many AD engines, such as `Zygote` and `Mooncake` support it.
With these rules, the AD engines can focus on handling the computational graph.

#align(center, canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-0.2, 1), (8.5, -3), stroke: (dash: "dashed"))
  content((4, 0), box(stroke: black, inset: 10pt, radius: 4pt, s[`ChainRulesCore.jl`]), name: "core")
  content((1.8, -2), box(stroke: black, inset: 10pt, radius: 4pt, s[`ChainRules.jl`]), name: "rules")
  content((6, -2), box(stroke: black, inset: 10pt, radius: 4pt, s[`User defined rules`]), name: "user")
  line("core", "rules", mark: (end: "straight"))
  line("core", "user", mark: (end: "straight"))

  content((4, -3.3), s[Rule set])

  rect((10, 1), (14, -3), stroke: (dash: "dashed"), name: "ad-engines")
  content((12, 0), box(stroke: black, inset: 10pt, radius: 4pt, s[`Mooncake.jl`]), name: "Mooncake")
  content((12, -1.5), box(stroke: black, inset: 10pt, radius: 4pt, s[`Zygote.jl`]), name: "Zygote")
  content((12, -2.5), s[`...`])
  content((12, -3.3), s[AD Engines])
  line("core", "ad-engines", mark: (end: "straight"))
}))

In the following example, we implement the symmetric eigen decomposition and its gradient function.
```julia
using DifferentiationInterface
import Mooncake, ChainRulesCore, LinearAlgebra
using ChainRulesCore: unthunk, NoTangent, @thunk, AbstractZero

# the forward function
function symeigen(A::AbstractMatrix)
    E, U = LinearAlgebra.eigen(A)
    E, Matrix(U)
end

# the backward function for the eigen decomposition
# References: Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
function symeigen_back(E::AbstractVector{T}, U, E̅, U̅; η=1e-40) where T
    all(x->x isa AbstractZero, (U̅, E̅)) && return NoTangent()
    η = T(η)
    if U̅ isa AbstractZero
        D = LinearAlgebra.Diagonal(E̅)
    else
        F = E .- E'
        F .= F./(F.^2 .+ η)
        dUU = U̅' * U .* F
        D = (dUU + dUU')/2
        if !(E̅ isa AbstractZero)
            D = D + LinearAlgebra.Diagonal(E̅)
        end
    end
    U * D * U'
end
```
In the backward function, the values of the output `E`, `U`, the adjoint of the output `E̅` and `U̅` are required. In `ChainRulesCore`, we can overload the `rrule` function to define the backward rule for the `symeigen` function.

```julia
# port the backward function to ChainRules
function ChainRulesCore.rrule(::typeof(symeigen), A)
    E, U = symeigen(A)
    function pullback(y̅)
        A̅ = @thunk symeigen_back(E, U, unthunk.(y̅)...)
        return (NoTangent(), A̅)
    end
    return (E, U), pullback
end
```
Here, the backward function (`pullback`) is implemented as a closure, which captures the values of the input `A`, and the output `E`, `U` automatically. It takes the adjoint of the output `(E̅, U̅)` as input and returns the adjoint of the function instance and function inputs.
Note the function instance can be parameterized and carry adjoints as well, i.e. a callable object. Here, the function instance is `symeigen` of type `typeof(symeigen)`, which is a constant that does not carry adjoints. So we return `NoTangent()` for the function instance.
In the body of the `pullback` function, we use `@thunk` to delay the evaluation of the function arguments, and `unthunk` to extract the value from the `Thunk` type. This technique is called the lazy evaluation, or evaluation on demand. Lazy evaluation is a powerful technique to avoid unnecessary computations in an AD engine.
The return value of `rrule` is a tuple of the function value and the pullback function. The function value is used in the forward computation and the pullback function is used in the backward pass. It indicates that the intermediate variables captured by the closure will not be deallocated until the backward pass is finished.

== Mooncake AD engine

We will introduce the Mooncake AD engine in the following example. It provides a convenient way to port the `rrule` to the AD engine.
```julia
# port the backward function to Mooncake
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(symeigen), Matrix{Float64}}
```
Note here, it is required to specify the type of the function inputs, since the above `rrule` is not valid for all input types.

```julia
# prepare a test function
function tfunc(A, target)
    E, U = symeigen(A)
    return sum(abs2, U[:, 1] - target)
end

# use the Mooncake AD engine
backend = DifferentiationInterface.AutoMooncake(; config=nothing)

# the function only takes one argument, so we wrap it in a tuple
wrapped(x) = tfunc(x...)

# pre-allocate the memory for the gradient, speed up the gradient computation
A = randn(Float64, 100, 100); A += A'
house = LinearAlgebra.normalize(vcat(zeros(20), 50 .+ (0:29), 80 .- (29:-1:0) , zeros(20)))
prep = DifferentiationInterface.prepare_gradient(wrapped, backend, (A, house))

# compute the gradient
g2 = DifferentiationInterface.gradient(wrapped, prep, backend, (A, house))
```

Mooncake also provides a convenient way to test the correctness of the rule by comparing with the finite difference method.
```julia
Mooncake.TestUtils.test_rule(Mooncake.Xoshiro(123), wrapped, (A, house); is_primitive=false)
```

== Differentiating linear algebra operations <differentiating-linear-algebra-operations>

In this section, we summarize the backward rules for common linear algebra operations. These rules are essential for implementing automatic differentiation in tensor network methods and other applications involving linear algebra.

#figure(
  table(
    columns: (auto, auto),
    [*Operation*], [*Backward Rule*],
    [Matrix-vector product\
    $y = A x$], 
    [$overline(A) = overline(y) x^T$\
    $overline(x) = A^T overline(y)$],
    
    [Matrix-matrix product\
    $C = A B$], 
    [$overline(A) = overline(C) B^T$\
    $overline(B) = A^T overline(C)$],
    
    [Matrix transpose\
    $B = A^T$], 
    [$overline(A) = overline(B)^T$],
    
    [Matrix inverse\
    $B = A^(-1)$], 
    [$overline(A) = -overline(B)^T overline(B) overline(A)^T$],
    
    [Matrix determinant\
    $y = det(A)$], 
    [$overline(A) = overline(y) y A^(-T)$],
    
    [Singular Value Decomposition\
    $A = U Sigma V^T$], 
    [Complex case @Wan2019\
    $overline(U) = overline(A)V Sigma^dagger + U hat(F)$\
    $overline(V) = overline(A)^T U Sigma + V hat(F)^T$\
    $overline(Sigma) = U^T overline(A)V$\
    where $hat(F)_(i j) = (overline(Sigma)_i Sigma_j - Sigma_i overline(Sigma)_j)/(Sigma_i^2 - Sigma_j^2)$ for $i != j$],
    
    [QR Decomposition\
    $A = Q R$], 
    [$overline(Q) = overline(A)R^T + Q overline(R)^T R^(-1) R^T - Q overline(R)^T R^(-1) R^T$\
    $overline(R) = Q^T overline(A)$],
    
    [Trace\
    $y = tr(A)$], 
    [$overline(A) = overline(y) I$],
    
    [Frobenius norm\
    $y = |A|_F$], 
    [$overline(A) = overline(y) A \/y$],
  ),
  caption: [Backward rules for common linear algebra operations. Here, $overline(x)$ denotes the adjoint of $x$, which is the derivative of the loss function with respect to $x$.]
)

For complex-valued operations, the backward rules become more intricate, especially for operations like SVD @Hubig2019 @Townsend2016. The complex-valued SVD backward rule shown in the table is derived in @Wan2019, which extends the real-valued case by properly handling the complex conjugation.

These backward rules form the foundation for implementing automatic differentiation in tensor network methods and other applications involving linear algebra operations. They allow us to efficiently compute gradients through complex computational graphs that include these operations.


== Differentiating implicit functions <differentiating-implicit-functions>

Considering a user-defined mapping $bold(F): RR^d times RR^n -> RR^d$ that encapsulates the optimality criteria of a given problem, an optimal solution, represented as $x(theta)$, is expected to satisfy the root condition of $bold(F)$ as follows:
$ bold(F)(x^*(theta), theta) = 0 $ <eq-Ffunction>

The function $x^*(theta): RR^n -> RR^d$ is implicitly defined. According to the implicit function theorem@Blondel2022, given a point $(x_0, theta_0)$ that satisfies $F(x_0, theta_0) = 0$ with a continuously differentiable function $bold(F)$, if the Jacobian $diff bold(F)/diff x$ evaluated at $(x_0, theta_0)$ forms a square invertible matrix, then there exists a function $x(dot)$ defined in a neighborhood of $theta_0$ such that $x^*(theta_0) = x_0$. Moreover, for all $theta$ in this neighborhood, it holds that $bold(F)(x^*(theta), theta) = 0$ and $(diff x^*)/(diff theta)$ exists. By applying the chain rule, the Jacobian $(diff x^*)/(diff theta)$ satisfies

$ (diff bold(F)(x^*, theta))/(diff x^*) (diff x^*)/(diff theta) + (diff bold(F)(x^*, theta))/(diff theta) = 0 $

Computing $diff x^* / diff theta$ entails solving the system of linear equations expressed as

$ underbrace((diff bold(F)(x^*, theta))/(diff x^*), "V" in RR^(d times d)) underbrace((diff x^*)/(diff theta), "J" in RR^(d times n)) = -underbrace((diff bold(F)(x^*, theta))/(diff theta), "P" in RR^(d times n)) $ <eq-implicit-linear-equation>

Therefore, the desired Jacobian is given by $J = V^(-1)P$. In many practical situations, explicitly constructing the Jacobian matrix is unnecessary. Instead, it suffices to perform left-multiplication or right-multiplication by $V$ and $P$. These operations are known as the vector-Jacobian product (VJP) and the Jacobian-vector product (JVP), respectively. They are valuable for determining $x(theta)$ using reverse-mode and forward-mode automatic differentiation (AD), respectively.

== Checkpointing <sec-checkpointing>
The main drawback of the reverse mode AD is the memory usage. The memory usage of the reverse mode AD is proportional to the number of intermediate variables, which scales linearly with the number of operations. The optimal checkpointing@Griewank2008 is a technique to reduce the memory usage of the reverse mode AD. It is a trade-off between the memory and the computational cost. The optimal checkpointing is a step towards solving the memory wall problem

Given the binomial function $eta(tau, delta) = ((tau + delta)!)/(tau!delta!)$, show that the following statement is true.
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k) $.

In @fig:checkpointing, we visualize the checkpointing scheme for calculating a program with 30 computational steps and 5 checkpoints. The red cells are the computed gradients, the gray cells are the checkpointed states, and the black cells and white faced cells are the created and removed pebbles in the current step.

The fundamental challenge of reverse mode AD is how to access the intermediate states in the reversed order of the computation. In many cases, it can be resolved by caching all intermediate states, but this may cost too much memory. The optimal checkpointing@Griewank2008 is a technique to reduce the memory usage of the reverse mode AD. By only caching logarithmically many intermediate states, the computational overhead is only logarithmic in the number of operations.

A _checkpoint_ is a snapshot of the computational state, which can be used to resume the computation. In the following, we consider a ordinary differential equation like program, in which case a checkpoint usually have a fixed size $S$, and the computating time is proportional to the number of steps $m$.

As illustrated in @fig:fig-checkpointing (b), the time-optimal checkpointing is to cache every state, which is the default behavior in most reverse mode AD engines. As we commented, this is memory-inefficient. While as illustrated in @fig:fig-checkpointing (a), the space optimal checkpointing is to cache only the initial state, denoted as $s_0$. This costs a quadratic overhead in the number of steps, i.e. $T ~ O(m^2)$. What we want in practice is a balanced checkpointing scheme, which is illustrated in @fig:fig-checkpointing (c), where time-space tradeoff is considered.

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  let dx = 0.5
  let states(loc, seq, prefix) = {
    for (i, s) in seq.enumerate(){
      circle((i * dx + loc.at(0), loc.at(1)), radius: 0.1, fill: (if (s == 1) {black} else {white}), name: prefix + str(i))
    }
  }
  states((0, 0), (1, 0, 0, 0, 0, 0), "s")
  bezier("s0", "s1.north", (0.5 * dx, 0.8), mark: (end: "straight"))
  bezier("s0", "s2.north", (0.5 * dx, 1.1), mark: (end: "straight"))
  bezier("s0", "s3.north", (0.5 * dx, 1.4), mark: (end: "straight"))
  bezier("s0", "s4.north", (0.5 * dx, 1.7), mark: (end: "straight"))
  bezier("s0", "s5.north", (0.5 * dx, 2), mark: (end: "straight"))

  content((1.25, -0.6), s[(a)])

  set-origin((4, 0))
  states((0, 0), (1, 1, 1, 1, 1, 0), "s")
  bezier("s0", "s5.north", (0.5 * dx, 2), mark: (end: "straight"))
  content((1.25, -0.6), s[(b)])

  set-origin((4, 0))
  states((0, 0), (1, 0, 0, 0, 1, 0), "s")
  bezier("s0", "s1.north", (0.5 * dx, 0.8), mark: (end: "straight"))
  bezier("s0", "s2.north", (0.5 * dx, 1.1), mark: (end: "straight"))
  bezier("s0", "s3.north", (0.5 * dx, 1.4), mark: (end: "straight"))
  bezier("s4", "s5.north", (4.5 * dx, 0.8), mark: (end: "straight"))
  for i in range(6){
    content((i * dx, -0.2), text(8pt)[$s_#i$])
  }
  content((1.25, -0.6), s[(c)])
}),
caption: [Checkpointing schemes: (a) space optimal, (b) time optimal and (c) balanced. The black and white circles represent the cached states and not cached states, respectively.]
) <fig:fig-checkpointing>

The optimal checkpointing even considers checkpointing for multiple passes. For example, in @fig:fig-checkpointing (c), after using $s_4$ to obtain $overline(s)_4$, the state $s_4$ can be removed from the cache since it is not used again. Then, in the next pass, during the computation of $s_3$, we can create a new checkpoint, e.g. the state $s_1$ can be cached. This does not increase the total memory usage, since we dropped state $s_4$ in the previous pass. The optimal strategy for the multiple-pass checkpointing is also known as the _Treeverse algorithm_@Griewank1992.
#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  let dx = 0.5
  let states(loc, seq, prefix) = {
    for (i, s) in seq.enumerate(){
      circle((i * dx + loc.at(0), loc.at(1)), radius: 0.1, fill: (if (s == 1) {black} else {if (s == 0) {white} else {gray.lighten(20%)}}), name: prefix + str(i))
    }
  }
  states((0, 0), (1, 1, 0, 0, 2, 0), "s")
  content((0.5, -0.7), text(11pt)[new], name: "new")
  line("new", "s1.south", mark: (end: "straight"))
  content((2, -0.2), text(11pt, red)[$times$])
  bezier("s1", "s2.north", (1.5 * dx, 0.8), mark: (end: "straight"))
  bezier("s0", "s3.north", (0.5 * dx, 1.4), mark: (end: "straight"))
  bezier("s4.north", "s5.north", (4.5 * dx, 0.8), mark: (end: "straight"))
  circle((2.25, 0), radius: 0.55, stroke: (dash: "dashed"))
  content((4, 0.5), text(11pt)[previous pass])
})
)

We consider the following question: Let $delta$ be the maximum number of checkpoints allowed in memory (not including the initial state), $tau$ be the number of passes, what is the maximum number of steps in the program?
Let us denote the maximum number of steps in the program given by the Treeverse algorithm as $eta(tau, delta)$. Then we have the following recurrence relation:
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k). $ <eq:treeverse-recurrence>
This is based on the following observations:
- In the next sweep, the number of sweeps allowed is decreased by 1, explaining $tau - 1$ on the right side.
- The current sweep devides the program into $delta + 1$ sectors with $delta$ checkpoints. In the next sweep, the number of allowed checkpoints for the $k$-th sector is $delta - k$, which is consistent with number of checkpoints behind the $k$-th sector.


Then we ask, what is the function that satisfies the recurrence relation in @eq:treeverse-recurrence?
The answer is the binomial function $ eta(tau, delta) = ((tau + delta)!)/(tau!delta!). $

#v(-3.3cm)
#figure(gap: -3.3cm, rotate(-90deg, canvas(length: 0.5cm, {
  visualize-treeverse("treeverse-30-5.json")
})), caption: [The checkpointing scheme for calculating a program with 30 computational steps and 5 checkpoints. The red cells are the computed gradients, the gray cells are the checkpointed states, and the black cells and white faced cells are the created and removed pebbles in the current step.]) <fig:checkpointing>


#algorithm({
  import algorithmic: *
  Function("Treeverse", ([$S$], [$overline(s_phi.alt)$], [$delta$], [$tau$], [$beta$], [$sigma$], [$phi.alt$]), {
    If($sigma > beta$, {
      Assign([$delta$], [$delta - 1$])
      Assign([$s$], [$S[beta] quad$ #CommentInline([Load initial state $s_beta$])])
      For($j = beta, beta+1, dots, sigma-1$, {
        Assign([$s_(j+1)$], [$f_j(s_j) quad$ #CommentInline([Compute $s_sigma$])])
      })
      Assign([$S[sigma]$], [$s_sigma$])
    })
    Comment[Recursively call Treeverse with optimal split point $kappa$ (binomial distribution)]
    While($tau > 0 "and" kappa = "mid"(delta, tau, sigma, phi.alt) < phi.alt$, {
      Assign([$overline(s_kappa)$], [treeverse($S$, $overline(s_phi.alt)$, $delta$, $tau$, $sigma$, $kappa$, $phi.alt$)])
      Assign([$tau$], [$tau - 1$])
      Assign([$phi.alt$], [$kappa$])
    })
    Assign([$overline(s_sigma)$], [$overline(f_sigma)(overline(s_(sigma+1)), s_sigma) quad$ #CommentInline([Use existing $s_sigma$ and $overline(s_phi.alt)$ to return gradient])])
    
    If($sigma > beta$, {
      Line([remove($S[sigma]$)  $quad$ #CommentInline([Remove $s_sigma$ from cached state set])])
    })
    Return[$overline(s_sigma)$]
  })

  Function("mid", ([$delta$], [$tau$], [$sigma$], [$phi.alt$]), {
    Comment[Select the binomial distribution split point]
    Assign([$kappa$], [$ceil((delta sigma + tau phi.alt)/(tau+delta))$])
    If($kappa >= phi.alt "and" delta > 0$, {
      Assign([$kappa$], [max($sigma+1$, $phi.alt-1$)])
    })
    Return[$kappa$]
  })
})


== Implementations

Automatic differentiation can be implemented in the sourceto-source transformation approach or the operator overloading approach.
Operator overloading AD exploits the operator overloading features of the programming language that the model is expressed in. Runtime libraries implement methods that overload operators such as linear algebra operations and mathematical functions to compute derivative values along with elementary computation.
Source-to-source transformation generates an output code that contains the original input code as well additional code to compute derivatives. It is typically faster than operator overloading AD as it can use compiler analysis such as activity analysis to optimize the generated code. Operator overloading AD is more robust as the tool does not have to handle esoteric syntactic and semantic programming language features itself.


To select a proper AD tool: source to source and operator overloading.

#figure(
  table(
    columns: (auto, auto, auto),
    [], [*Source to source*], [*Operator overloading*],
    [Primitive], [basic scalar operations], [tensor operations],
    [Application], 
    align(left)[- physics simulation], 
    align(left)[- machine learning],
    [Advantage],
    align(left)[
      - correctness
      - handles effective code
      - works on generic code
    ],
    align(left)[
      - fast tensor operations
      - extensible
    ],
    [Package],
    align(left)[
      - Tapenade@Hascoet2013
      - Enzyme@Moses2021
    ],
    align(left)[
      - Jax@Jax2018
      - PyTorch@Paszke2019
    ]
  ),
  caption: "Most of the packages listed above supports both forward and backward mode AD."
)



== Adjoint State Method

The Adjoint State Method@Plessix2006 @Chen2018 is a specific method for reverse propagation of ordinary differential equations. In research, it has been found that the reverse propagation of the derivative of the integration process is also an integration process, but in the opposite direction. Therefore, by constructing an extended function that can simultaneously trace the function value and backpropagate the derivative, the calculation of the derivative is completed in the form of inverse integration of the extended function, as shown in Algorithm 1. The description of this algorithm comes from @Chen2018, where detailed derivation can be found. Here, the symbols in the original algorithm have been replaced for better understanding. The local derivatives $(diff q)/(diff s)$, $(diff q)/(diff theta)$, and $(diff cal(L))/(diff s_n)$ in the algorithm can be manually derived or implemented using other automatic differentiation libraries. This method ensures strict gradients when the integrator is strictly reversible, but when the integration error in the reverse integration of the integrator cannot be ignored, additional processing is required to ensure that the error is within a controllable range, which will be discussed in subsequent examples.

#figure(
align(left, algorithm({
  import algorithmic: *
  Function("Adjoint-State-Method", ([$s_n$], [$s_0$], [$theta$], [$t_0$], [$t_n$], [$cal(L)$]), {
    Comment[Define the augmented dynamics function]
    Function("aug_dynamics", ([$s$], [$a$], [$theta$]), {
      Assign([$q$], [$f(s, t, theta)$])
      Return[$q$, $-a^T (diff q)/(diff s)$, $-a^T (diff q)/(diff theta)$]
    })
    Comment[Compute the initial state for the augmented dynamics function]
    Assign([$S_n$], [$(s_n, (diff cal(L))/(diff s_n), 0)$])
    Comment[Perform reverse integration of the augmented dynamics]
    Assign([$(s_0, (diff cal(L))/(diff s_0), (diff cal(L))/(diff theta))$], CallInline("ODESolve", (smallcaps("aug_dynamics"), [$S_n$], [$theta$], [$t_n$], [$t_0$]).join(", ")))
    Return[$(diff cal(L))/(diff s_0)$, $(diff cal(L))/(diff theta)$]
  })
})),
caption: [The continuous adjoint state method])

#figure(
  canvas({}),
  caption: [
    Using (a) checkpointing scheme and (b) reverse computing scheme to avoid caching all intermediate states. The black arrows are regular forward computing, red arrows are gradient back propagation, and blue arrows are reverse computing. The numbers above the arrows are the execution order.
    Black and white circles represent cached states and not cached states (or those states deallocated in reverse computing) respectively.
  ]
)

// = Applications

// Differential programming tensor networks @Liao2019 @Francuz2023
== Applications of Automatic Differentiation in Physical Simulations

This chapter contains two case studies: one is solving the derivatives of the Lorenz system with fewer parameters using forward automatic differentiation and the adjoint state method, and the other is solving the derivatives of a seismology simulation with many steps and huge memory consumption using backward differentiation based on the optimal checkpoint algorithm and reversible computing.

== Solving the Lorenz Equations
The Lorenz system@Lorenz1963 is a classic model for studying chaos, describing dynamics defined in three-dimensional space
$ 
(D x)/(D t) &= sigma(y - x),\
(D y)/(D t) &= x(rho -z) - y,\
(D z)/(D t) &= x y - beta z.
$
where $sigma$, $rho$, and $beta$ are three control parameters. The time evolution curve of this system is shown in Figure 2 (b). When $rho>1$, the system has two attractors@Hirsch2012, but only when $rho < sigma (sigma + beta + 3)/(sigma - beta - 1)$ will the particles stably orbit around one of the attractors as shown by the orange curve in Figure (b). At this time, the system is relatively stable and exhibits less sensitivity to initial values. The derivatives of the final position coordinates with respect to the control parameters and initial coordinates reflect the sensitivity of the final state to the control parameters and initial positions, to some extent indicating the occurrence of chaos in the system.

In numerical simulations, we use the 4th-order Runge-Kutta method for time integration to obtain the final position, with the initial position fixed at $(x_0,y_0,z_0) = (1, 0, 0)$, the control parameter $beta=8/3$, the integration time interval $[0,T=30]$, and the integration step size $3 times 10^(-3)$.

Since this process contains only 6 parameters, including the three coordinates of the initial position $(x_0, y_0, z_0)$ and the three control parameters $(sigma, rho,beta)$, using the forward automatic differentiation tool ForwardDiff@Revels2016 for differentiation has a great advantage over backward automatic differentiation. We plot the average absolute value of the derivatives with respect to the initial $rho$ and $sigma$ in Figure 2. It can be seen that only below the theoretical prediction black line will there be smaller derivatives, indicating that the system dynamics under stable attractor parameters indeed have low dependence on initial values.

== Differentiation of the Wave Propagation Equation
Consider the propagation of the wave function $u(x_1, x_2, t)$ in a non-uniform two-dimensional medium governed by the following equation
$ 
cases(
  (partial^2 u)/(partial t^2) - nabla dot(c^2 nabla u) = f & t>0,
  u = u_0 & t=0,
  (partial u)/(partial t) = v_0 & t=0
)
$
where $c$ is the wave propagation speed in the medium. The Perfectly Matched Layer (PML)@Berenger1994@Roden2000@Martin2008 is an accurate and reliable method for simulating wave motion in a medium. To simulate this dynamics in a finite size, the PML method introduces an absorbing layer to prevent boundary reflection effects. By introducing auxiliary fields and discretizing space and time, the above equation can be transformed into the following numerical computation process
$
u^(n+1)_(i,j) approx &(Delta t^2)/(1+(zeta_(1i)+zeta_(2j))Delta t/2) (
    (2-zeta_(1i)zeta_(2j))u_(i,j)^n - (1-(zeta_(1i)+zeta_(2j))Delta t/2)/(Delta t^2)u^(n-1)_(i,j)\ &+ c_(i,j)^2(u_(i+1,j)^n-2u_(i,j)^n+u_(i-1,j)^n)/(Delta x^2) + c_(i,j)^2(u_(i,j+1)^n-2u_(i,j)^n+u_(i,j-1)^n)/(Delta y^2)\
    &+ ((phi.alt_x)_(i+1,j)-(phi.alt_x)_(i-1,j))/(2Delta x) + ((phi.alt_y)_(i,j+1)-(phi.alt_y)_(i,j-1))/(2Delta y) ),\
  (phi.alt_x)_(i,j)^(n+1) &= (1-Delta t zeta_(1i))(phi.alt_x)_(i,j)^n + Delta t c_(i,j)^2 (zeta_(1i)-zeta_(2j))(u_(i+1,j)-u_(i-1,j))/(2Delta x),\
  (phi.alt_y)_(i,j)^(n+1) &= (1-Delta t zeta_(2j))(phi.alt_y)_(i,j)^n + Delta t c_(i,j)^2 (zeta_(2j)-zeta_(1i))(u_(i,j+1)-u_(i,j-1))/(2Delta y)
$

The first term here is an approximation because it ignores the contribution of the spatial gradient term of the medium propagation speed $c$ in the original equation. $zeta_1$ and $zeta_2$ are the attenuation coefficients in the $x$ and $y$ directions, respectively, and $phi.alt_x$ and $phi.alt_y$ are the $x$ and $y$ components of the introduced auxiliary fields, respectively. $Delta x$, $Delta y$, and $Delta t$ are the spatial and temporal discretization parameters, respectively. The detailed derivation of this equation can be found in the literature@Grote2010.

Automatic differentiation of PML simulations has important applications in seismology@Zhu2021, and it has long been recognized that checkpoint schemes can be used in seismic wave simulations@Symes2007 to greatly reduce the memory requirements for backtracking intermediate states.

#figure(canvas({}),
  caption: [The time and space overhead of applying the Bennett algorithm and the Treeverse algorithm to the differentiation of the PML solving process. The numbers marked in the figure are the ratios of the actual forward running steps to the original simulation steps ($10^4$). The total time on the vertical axis is the sum of the forward computation and backpropagation time, and the value on the horizontal axis represents the number of checkpoints or the maximum number of states in reversible computing. In the Bennett algorithm, the number of backward steps is the same as the number of forward steps, while in the Treeverse algorithm, the number of backpropagation steps is fixed at $10^4$.]
)

In numerical simulations, we simulated the PML equation on a $1000 times 1000$ two-dimensional grid using double-precision floating-point numbers, with each state storing 4 matrices $s_n = {u^(n-1), u^n, phi.alt_x^n, phi.alt_y^n}$, occupying 32MB of storage space.

Although forward automatic differentiation can differentiate this program with only a constant multiple of space overhead, the linear increase in time complexity due to the $10^6$ parameters of the propagation speed $c$ is unacceptable. At the same time, if backward automatic differentiation is performed on this program without any memory optimization, integrating $10^4$ steps requires at least 320G of storage space, far exceeding the storage capacity of ordinary GPUs.

At this point, we need to use the Bennett algorithm and the Treeverse algorithm described in the appendix to save the cache for backward automatic differentiation. Figure 5 shows the relationship between the actual program's time and space on the GPU using these two time-space tradeoff schemes, with the computing device being an Nvidia Tesla V100.

In the pure reversible computing implementation of the Bennett algorithm, the gradient computation part increases with the number of forward computation steps, and the additional overhead is almost consistent with the theoretical model.

The Bennett algorithm is the optimal time-space tradeoff strategy in the sense of reversible computing, but it is not optimal for ordinary hardware.

The Treeverse+NiLang scheme refers to using reversible computing to handle the differentiation of single-step operations, while using the Treeverse algorithm to handle the differentiation between steps. Here, as the number of checkpoints decreases, the reduction in computation time is not significant. This is because the increased computation time is for forward computation, and here the single-step backward gradient computation time is more than twenty times that of the forward time, so even with only 5 checkpoints, the additional time increase is less than one time.

The reason why the single-step gradient computation is much slower than the forward computation is that when using NiLang to differentiate parallel GPU kernel functions, it is necessary to avoid shared variable reads to prevent the program from simultaneously updating the gradient of the same memory block in parallel during backward computation, which brings a lot of performance loss. In contrast, on a single-threaded CPU, the time difference between forward and backward single-step operations is within four times.

In addition, although the Treeverse algorithm can achieve efficient time and space tradeoff, it cannot be directly used to differentiate GPU kernel functions due to the need for global stack management in system memory. Reversible computing is very suitable for differentiating such nonlinear and somewhat reversible programs, which is why we choose to use reversible computing to differentiate single-step operations to avoid the trouble of manually differentiating GPU kernel functions.




= Appendix: Differentiating linear algebra operations <differentiating-linear-algebra-operations>


== Notations

We derived the following useful relations:
$ tr[A(C compose B)] = sum A^T compose C compose B = tr((C compose A^T)^T B) = tr(C^T compose A)B $ <eq:tr_compose>

$ (C compose A)^T = C^T compose A^T $ <eq:transpose_compose>

Let $cal(L)$ be a real function of a complex variable $x$, $ (diff cal(L))/(diff x^*) = ((diff cal(L))/(diff x))^* $ <eq:diff_complex>



== Matrix multiplication <matrix-multiplication>

Matrix multiplication $C = A B$, where $A in CC^(m times n)$ and $B in CC^(n times p)$.
$ cases(
  overline(A) &= overline(C) B^dagger,
  overline(B) &= A^dagger overline(C)
) $

Let us introduce a small perturbation $delta A$ on $A$ and $delta B$ on $B$,

$ delta C = delta A B + A delta B $

$ delta cal(L) = tr(delta C^T overline(C)) = 
tr(delta A^T overline(A)) + tr(delta B^T overline(B)) $

It is easy to see
$ delta L = tr((delta A B)^T overline(C)) + tr((A delta B)^T overline(C)) = 
tr(delta A^T overline(A)) + tr(delta B^T overline(B)) $

We have the backward rules for matrix multiplication as
$ 
cases(
  overline(A) = overline(C)B^T,
  overline(B) = A^T overline(C)
)
$

== QR decomposition <qr-decomposition>
QR decomposition.
Let $A$ be a matrix with full row or column rank, the QR decomposition is defined as
$ A = Q R $
with $Q^dagger Q = bb(I)$, so that $d Q^dagger Q + Q^dagger d Q = 0$. $R$ is a complex upper triangular matrix, with diagonal elements being real.

$
  overline(A) = overline(Q) + Q "copyltu"(M)R^(-dagger),
$
where $M = R^(-1)overline(R)^dagger - overline(Q)^dagger Q$.
The $"copyltu"$ operator takes the conjugate when copying elements to the upper triangular part.

#strong([For some special cases:])

(1) For $A in CC^(m times n), m<n$, we choose $R^(-1)$ such that $R R^(-1)=bb(I)$.

$R^r$ can be obtained easily by applying the same column transformations to both $R$ and $I_n$ until $A$ is transformed into $(I_m,0)$. $R^r$ satisfies the following: if we denote the position of the first nonzero element in the $i_(t h)$ row of $R$ as $1<=i_1<..<i_m$, then the $(i,j)$ element of $R^r$ can be nonzero only when:
$
  &i=i_k in {i_1,..,i_m}, j>=k
$

Furthermore, it is straightforward to prove that such $R^r$ is unique.

(2) For QR decomposition with pivoting, we simply assume that the permutation matrix $P$ is a constant matrix.

The backward rules for QR decomposition are derived in multiple references, including @Hubig2019 and @Liao2019. To derive the backward rules, we first consider differentiating the QR decomposition
@Seeger2017, @Liao2019

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

$ U compose (d C + d C^dagger) = d R R^(-1) $

where $U$ is a mask operator that its element value is $1$ for upper triangular part, $0.5$ for diagonal part and $0$ for lower triangular part. One should also notice here both $R$ and $d R$ has real diagonal parts, as well as the product $d R R^(-1)$.

We have

$ 
  d cal(L) &= tr[overline(Q)^dagger d Q + overline(R)^dagger d R + "h.c."],\
  &= tr[overline(Q)^dagger d A R^(-1) - overline(Q)^dagger Q d R R^(-1) + overline(R)^dagger d R + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + R^(-1)(-overline(Q)^dagger Q + R overline(R)^dagger)d R + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + R^(-1)M d R + "h.c."]
$

here, $M = R overline(R)^dagger - overline(Q)^dagger Q$. Plug in $d R$ we have

$ 
  d cal(L) &= tr[R^(-1)overline(Q)^dagger d A + M[U compose (d C + d C^dagger)] + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L)(d C + d C^dagger) + "h.c."] #h(2em),\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L)(d C + d C^dagger) + (M compose L)^dagger (d C + d C^dagger)],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L + "h.c.")d C + "h.c."],\
  &= tr[R^(-1)overline(Q)^dagger d A + (M compose L + "h.c.")Q^dagger d A R^(-1)] + "h.c."
$

where $L = U^dagger = 1-U$ is the mask of lower triangular part of a matrix.
In the second line, we have used @eq:tr_compose.

$
  overline(A)^dagger &= R^(-1)[overline(Q)^dagger + (M compose L + "h.c.")Q^dagger],\
  overline(A) &= [overline(Q) + Q "copyltu"(M)]R^(-dagger),\
  &= [overline(Q) + Q "copyltu"(M)]R^(-dagger)
$

Here, the $"copyltu"$ takes conjugate when copying elements to upper triangular part.

== Singular value decomposition <singular-value-decomposition>

=== References
- SVD @Hubig2019, @Townsend2016, @Giles2008
- Complex SVD @Wan2019
- Truncated SVD @Francuz2023

Complex valued singular value decomposition
$
&A = U S V^dagger,\ &V^dagger V = I,\ &U^dagger U = I,\ &S = "diag"(s_1, ..., s_n),
$ <eq:svd>
where the input $A$ is a complex matrix, the outputs $U$ is a unitary matrix, $S$ is a real diagonal matrix and $V$ is a unitary matrix. We also apply an extra constraint that the loss function $cal(L)$ is real and is invariant under the gauge transformation: $U arrow.r U Lambda$, $V arrow.r V Lambda$, where $Lambda$ is defined as $"diag"(e^(i phi_1), ..., e^(i phi_n))$.


$
  overline(A) = &U(J + J^dagger) S V^dagger + (I-U U^dagger)overline(U)S^(-1)V^dagger,\
  &+ U S(K + K^dagger)V^dagger + U S^(-1) overline(V)^dagger (I - V V^dagger),\
  &+ U (overline(S) compose I) V^dagger,\
  &+ 1/2 U (S^(-1) compose(U^dagger overline(U))-h.c.)V^dagger
$ <eq:svd_loss_diff_full>
where $J=F compose(U^dagger overline(U))$, $K=F compose(V^dagger overline(V))$ and $F_(i j) = cases( 1/(s_j^2-s_i^2) \, &i!=j, 0\, &i=j)$.

We start with the following two relation
$
  2 delta cal(L) = tr[overline(A)^dagger delta A + h.c.] = tr[overline(U)^dagger delta U + overline(V)^dagger delta V + h.c.] + 2tr[overline(S) delta S]
$ <eq:loss_diff>
//where we have used @eq:diff_complex.

$
delta A = delta U S V^dagger + U delta S V^dagger + U S delta V^dagger
$ <eq:svd_diff>
//The clue is to resolve the right hand side of @eq:loss_diff into the form of $tr[f(A, overline(U), overline(V), overline(S)) delta A]$, then we will have $overline(A) = f(A, overline(U), overline(V), overline(S))^dagger$ as $delta A$ is arbitrary.

We first sandwich @eq:svd_diff between $U^dagger$ and $V$ and obtain
$
U^dagger delta A V &= U^dagger delta U S + delta S + S delta V^dagger V.
$
Then we denote $delta C=U^dagger delta U$, $delta D = delta V^dagger V$ and $delta P = U^dagger delta A V$,
then by using the second and third line in @eq:svd, we have $d U$ and $d V$ are skew-symmetric, i.e.

$ cases(
  delta C^dagger + delta C = 0,
  delta D^dagger + delta D = 0
) $ <eq:svd_delta_c_d>

We can simplify @eq:svd_diff as

$ delta P = delta C S + delta S + S delta D. $ <eq:svd_delta_p>

Since $delta C$ and $delta D$ are skew-symmetric, they must have zero real part in diagonal elements. It immediately follows that
$
delta S = Re[I compose delta P] = I compose (U^dagger delta A V + h.c.)/2.
$ <eq:svd_delta_s>

Let us denote the complement of $I$ as $overline(I) = 1-I$. We have
$
cases(
  overline(I) compose delta C = (overline(I) compose delta P) S^(-1) - S delta D S^(-1),
  overline(I) compose delta D = S^(-1) (overline(I) compose delta P) - S^(-1) delta C S,
  I compose (delta C + delta D) = i Im[I compose delta P] S^(-1)
)
$
The last line is for determining the imaginary diagonal part of $delta C$ and $delta D$, which can not be determined from the first two lines.
Combining with @eq:svd_delta_c_d, we have

$
&cases(
  S (overline(I) compose delta P) + (overline(I) compose delta P)^dagger S &= S^2 (overline(I) compose delta D)-delta D S^2,
  (overline(I) compose delta P) S + S (overline(I) compose delta P)^dagger &= (overline(I) compose delta C) S^2-S^2 delta C
),\ 
arrow.double.r &cases(
    overline(I) compose delta D = -F compose (S delta P + delta P^dagger S),
    overline(I) compose delta C = F compose (delta P S + S delta P^dagger),
    I compose (delta C + delta D) = S^(-1) compose (delta P - delta P^dagger)/2
)
$ <eq:svd_delta_c_d_p>
where $ F_(i j) = cases(1/(s_j^2-s_i^2)\, &i != j, 0\, &i = j). $ From top to bottom, we also need to consider the contribution from the diagonal imaginary parts of $delta P$.
It is important to notice here, the imaginary diagonal parts of $delta P$ is impossible to be determined from the above equation, since they are cancelled out.
Hence, we still need the extra constraints, which is the gauge invariance of the loss function.

To wrap up, we have

$
  tr[overline(A)^dagger delta A + h.c.] &= tr[overline(U)^dagger delta U + overline(V)^dagger delta V + overline(S) delta S + h.c.]\
  &= tr[overline(U)^dagger U delta C + V S^(-1) overline(U)^dagger (I-U U^dagger) delta A + h.c.]\
  &quad - tr[overline(V)^dagger V delta D - U S^(-1) overline(V)^dagger (I-V V^dagger) delta A^dagger + h.c.]\
  &quad + tr[(overline(S) compose I) (U^dagger delta A V + h.c.)]
$ <eq:svd_loss_diff>
where we have used
$
delta U &= (U U^dagger)delta U + (I-U U^dagger)delta U = U delta C + (I-U U^dagger)delta A V S^(-1),\
delta V &= (V V^dagger)delta V + (I-V V^dagger)delta V = -V delta D + (I-V V^dagger)delta A^dagger U S^(-1).
$
The second term in the first and second line can be derived by multiplying @eq:svd_diff by $(I - U U^dagger)$ on the left and $(I - V V^dagger)$ on the right respectively.
We first consider the off-diagonal terms in @eq:svd_delta_c_d_p, and plug them into @eq:svd_loss_diff, we have
$
tr[overline(U)^dagger U (overline(I) compose delta C) + h.c.] &= tr[overline(U)^dagger U (F compose (delta P S +  S delta P^dagger)) + h.c.]\
&= tr[V S (J + J^dagger) U^dagger delta A + h.c.]
$
where $J = F compose (U^dagger overline(U))$, which has diagonal elements being all zeros.
Similarly, we have
$
-tr[overline(V)^dagger V (overline(I) compose delta D) + h.c.] &= tr[V (K + K^dagger) S U^dagger delta A + h.c.]
$
where $K = F compose (V^dagger overline(V))$.

$ tr[(S^(-1)  compose (overline(U)^dagger U - U^dagger overline(U))/2) U^dagger delta A V + h.c.] $

Now lets consider the diagonal terms in @eq:svd_delta_c_d_p, and plug them into @eq:svd_loss_diff, we have
$
&tr[overline(U)^dagger U (I compose delta C) - V^dagger V (I compose delta D) + h.c.]\
&= tr[(I compose (overline(U)^dagger U - h.c.)) delta C - (I compose (overline(V)^dagger V - h.c.)) delta D]\
$ <eq:svd_loss_diff_diag>

At a first glance, it is not sufficient to derive $delta C$ and $delta D$ from $delta P$, but consider there is still an constraint not used, *the loss must be gauge invariant*, which means

$ cal(L)(U Lambda, S, V Lambda) $

Should be independent of the choice of gauge $Lambda$, which is defined as $"diag"(e^(i phi_1), ..., e^(i phi_n))$.
Now consider a infinitesimal gauge transformation $U arrow.r U (I + i delta phi)$ and $V arrow.r V (I + i delta phi)$, where $delta phi = "diag"(delta phi_1, ..., delta phi_n)$.
When reflecting this change on the loss function, we have

$
  2 delta cal(L) = tr[overline(U)^dagger U i delta phi + overline(V)^dagger V i delta phi + "h.c."] = 0
$
which is equivalent to
$ (I compose (overline(U)^dagger U - h.c.)) + (I compose (overline(V)^dagger V - h.c.)) = 0. $

Inserting this constraint into @eq:svd_loss_diff_diag, we have
$
tr[(I compose (overline(U)^dagger U - h.c.)) (delta C + delta D)]
$
Using @eq:svd_delta_c_d_p, we have
$
&tr[(overline(U)^dagger U - h.c.)(S^(-1) compose (delta P - delta P^dagger)/2)]\
= &tr[(S^(-1) compose (overline(U)^dagger U - h.c.)/2) U^dagger delta A V + h.c.]\
$


Collecting all terms, we have
$
  tr[overline(A)^dagger delta A + h.c.] &=
  tr[V S (J + J^dagger) U^dagger delta A + h.c.]\
  &quad + tr[V S^(-1) overline(U)^dagger (I-U U^dagger) delta A + h.c.]\
  &quad + tr[V (K + K^dagger) S U^dagger delta A + h.c.]\
  &quad + tr[U S^(-1) overline(V)^dagger (I-V V^dagger) delta A^dagger + h.c.]\
  &quad + tr[(S^(-1) compose (overline(U)^dagger U - h.c.)/2) U^dagger delta A V + h.c.]\
  &quad + tr[(overline(S) compose I) (U^dagger delta A V) + h.c.]
$

Collecting all terms associated with $delta A$, we have
$
  overline(A) &= U (J + J^dagger) S V^dagger && quad triangle.small.r "from " overline(U)\
  &quad + (I-U U^dagger) overline(U) S^(-1) V && quad triangle.small.r "if" U "is not full rank"\
  &quad + U S (K + K^dagger) V^dagger && quad triangle.small.r "from " overline(V)\
  &quad + U S^(-1) overline(V)^dagger (I-V V^dagger) && quad triangle.small.r "if" V "is not full rank"\
  &quad + U (S^(-1) compose (U^dagger overline(U) - h.c.)/2) V^dagger  && quad triangle.small.r "from gauge"\
  &quad + U (overline(S) compose I) V^dagger,   && quad triangle.small.r "from " overline(S)
$
which is exactly the same as @eq:svd_loss_diff_full.

#bibliography("refs.bib")