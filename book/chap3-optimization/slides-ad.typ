#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": canvas, draw, tree, vector, decorations, coordinate
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/algorithmic:1.0.3"
#import "@preview/ctheorems:1.1.3": *
#import "images/treeverse.typ": visualize-treeverse
#import algorithmic: algorithm
#set math.mat(row-gap: 0.1em, column-gap: 0.7em)

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)
#let proposition = thmbox("proposition", "Proposition", inset: (x: 1.2em, top: 1em), base: none)
#let theorem = thmbox("theorem", "Theorem", base: none)
#let proof = thmproof("proof", "Proof")

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Gradient-based optimization and automatic differentiation],
    subtitle: [],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#let bob(loc, rescale: 1, flip: false, label: none, words: none) = {
  import draw: *
  let r = 0.4 * rescale
  let xr = if flip { -r } else { r }
  circle(loc, radius: (0.8 * r, r), name: "bob")
  circle((rel: (xr * 0.4, 0.2 * r), to: loc), radius: (0.2 * r, 0.18 * r), name: "eye", stroke: none, fill: black)
  line((rel: (-1.5 * xr, -r), to: "bob"), (rel: (-0.6 * xr, -3.5 * r), to: "bob"), (rel: (0.7 * xr, -3.5 * r), to: "bob"), (rel: (1.2 * xr, -r), to: "bob"), stroke: (paint: black, thickness: 1pt), name: "line1", close: true)
  line((anchor: 31%, name: "line1"), (loc.at(0) - 0.5 * xr, loc.at(1) - 5 * r))
  line((anchor: 40%, name: "line1"), (loc.at(0) + 0.5 * xr, loc.at(1) - 5 * r))
  line((anchor: 20%, name: "line1"), (loc.at(0) + 0.5 * xr, loc.at(1) - 2 * r))
  line((anchor: 59%, name: "line1"), (loc.at(0) + 2 * xr, loc.at(1) - 2 * r))
  if label != none {
    content((loc.at(0), loc.at(1) - 1.5 * r), label)
  }
  if words != none {
    content((loc.at(0) + 10 * xr, loc.at(1) - 1.5 * r), box(width: rescale * 270pt, words))
  }
}

#title-slide()
#outline-slide()

= Gradient-based optimization

= Finite difference
==
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

== *Example: deriving the $4$-th order central finite difference*

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

== Problem of finite difference

- Not accurate enough.

== Computational graph
#figure((
  canvas(length: 2cm, {
    import draw: *
    let s(x) = text(16pt, x)
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

    content((1, -1.5), [Computational graph: nodes are functions, lines are data.])
  })
))


Q: How to represent `y = sin(x)^2 + cos(x)^2`?

= Forward mode autodiff
- _Forward mode AD_ presumes the scalar input. Given a program with scalar input $t$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _derivatives_ as $dot(bold(y)_i) = (partial bold(y)_i)/(partial t)$.
The _forward rule_ defines the transition between $bold(y)_i$ and $bold(y)_(i+1)$
$
dot(bold(y))_(i+1) = (partial bold(y)_(i+1))/(partial bold(y)_i) dot(bold(y))_i.
$
- _Remark_: In an automatic differentiation engine, the Jacobian matrix $(partial bold(y)_(i+1))/(partial bold(y)_i)$ is almost never computed explicitly in memory as it can be costly. Instead, the forward mode automatic differentiation can be implemented by overloading the function $f_i$ as
$
f_i^("forward"): (bold(y)_i, dot(bold(y))_i) arrow.bar (bold(y)_(i+1), (partial bold(y)_(i+1))/(partial bold(y)_i) dot(bold(y))_i)
$

== Example
In Julia, the package `ForwardDiff.jl`@Revels2016 provides a function `gradient` to compute the gradient of a simple function $f(theta, r) = r cos theta sin theta$.
```julia
using ForwardDiff

theta, r = π/4, 2.0
x = [theta, r]
f(x) = x[2] * cos(x[1]) * sin(x[1])
ForwardDiff.gradient(f, x) # [0.0, 0.5]
```

== Problem of forward mode AD
- Overhead is linear to the input size (the same as finite difference).

= Reverse mode autodiff
- The _reverse mode AD_ presumes a scalar output $cal(L)$, or the loss function. Given a program with scalar output $cal(L)$, we can denote the intermediate variables of the program as $bold(y)_i$, and their _adjoints_ as $overline(bold(y))_i = (partial cal(L))/(partial bold(y)_i)$.
- The _backward rule_ defines the transition between $overline(bold(y))_(i+1)$ and $overline(bold(y))_i$
$
overline(bold(y))_i = overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i).
$
We define the backward function $overline(f)_i$ as
$ overline(f)_i: ("TAPE", overline(bold(y))_(i+1)) arrow.bar ("TAPE", overline(bold(y))_(i+1) (partial bold(y)_(i+1))/(partial bold(y)_i)), $

== Reverse mode AD

#figure((
  canvas(length: 2cm, {
    import draw: *
    let s(x) = text(16pt, x)
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
)) <fig:computational_graph>


==  Example
In Julia, the package `Enzyme.jl`@Moses2021 provides a state of the art source code transformation based AD engine.
```julia
using Enzyme

gval = zero(x)
_, fval = Enzyme.autodiff(ReverseWithPrimal, f, Active, Duplicated(x, gval))
fval, gval  # 1.0, [0.0, 0.5]
```
- The first parameter `ReverseWithPrimal` means return both the gradient and the (primal) function value. If a scalar variable carries gradient, it should be declared as `Active`, if not, it should be declared as `Const`.
- For mutable objects like arrays with gradients, it should be declared as `Duplicated`, with an extra field to store gradients. Here, the output is a scalar with gradient, hence the thrid field is `Active`, the input is a vector with gradient, hence the last field is a `Duplicated` vector with a mutable gradient field. After calling `Enzyme.autodiff`, the gradient field is changed inplace.


== Obtaining Hessian

- Hessian can be computed by taking the Jacobian of the gradient.

== Hessian vector product
In many algorithms, we are only interested in Hessian vector product:
- stochastic reconfiguration @Sorella1998 (or the Dirac-Frenkel variational principle @Raab2000)
- Newton's method
$
  (partial^2 f)/(partial x^2)v = (partial ((partial f)/(partial x)v))/(partial x)
$ <eq:hvp>
- Obtaining the full Hessian matrix: $O(n)$ overhead ($n$ is the number of parameters).
- Hessian vector product: $O(1)$ overhead.

== Computational graph
#figure(canvas(length: 2cm, {
  import draw: *
  let s(x) = text(16pt, x)
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
  content((rel: (0.2, 0.2), to: "l3.mid"), s[$(partial f)/(partial x)$])
  content((rel: (0.2, -0.1), to: "l4.mid"), s[$v$])
  content((rel: (0.3, 0.0), to: "l5.end"), s[$(partial f)/(partial x)v$])
})) <fig:jacobian-vector>


== Example
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
- Treating real and imaginary parts as independent variables.
    $
    overline(z) = overline(x) + i overline(y).
    $
    If we change $z$ by a small amount $delta z = delta x + i delta y$, the loss function $cal(L)$ will change by
    $ delta cal(L) = (overline(z)^* delta z + h.c.)\/2 = overline(x) delta x + overline(y) delta y. $
- Gradient $!=$ Adjoint, because $CC arrow.r RR$ mapping can not be holomorphic!


== Example: Differentiating a complex valued function

`Enzyme` can be used to compute the gradient of a complex valued function.
```julia
fc(z) = abs2(z)
# a fixed input to use for testing
z = 3.1 + 2.7im
Enzyme.autodiff(Reverse, fc, Active, Active(z))[1][1]  # 6.2 + 5.4im
```
In this example, the input is a complex number $z$, and the output is a real number $|z|^2$. The gradient is $2 z$.

== Operator overloading based AD

- Source to source (e.g. Enzyme): based on source code transformation, has a finite scalar rule set
- Operator overloading based: extensible rule set, can fully utilize BLAS.

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

== Example: Backward rule for Symmetric Eigen Decomposition
In the following example, we implement the symmetric eigen decomposition and its gradient function.
#box(text(16pt, [```julia
using DifferentiationInterface
import Mooncake, ChainRulesCore, LinearAlgebra
using ChainRulesCore: unthunk, NoTangent, @thunk, AbstractZero

# the forward function
function symeigen(A::AbstractMatrix)
    E, U = LinearAlgebra.eigen(A)
    E, Matrix(U)
end
```]))

== Customizing the backward function
#box(text(16pt, [```julia
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
```]
))

== Customizing the backward function
#box(text(16pt, [```julia
# port the backward function to ChainRules
function ChainRulesCore.rrule(::typeof(symeigen), A)
    E, U = symeigen(A)
    function pullback(y̅)
        A̅ = @thunk symeigen_back(E, U, unthunk.(y̅)...)
        return (NoTangent(), A̅)
    end
    return (E, U), pullback
end
```]))

== Mooncake AD engine

You can use the `Mooncake.@from_rrule` macro to port the `rrule` to the Mooncake AD engine.
#box(text(16pt, [```julia
# port the backward function to Mooncake
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(symeigen), Matrix{Float64}}
```]))

==
#box(text(16pt, [```julia
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
```]))

== Test your rule
Mooncake also provides a convenient way to test the correctness of the rule by comparing with the finite difference method.
```julia
Mooncake.TestUtils.test_rule(Mooncake.Xoshiro(123), wrapped, (A, house); is_primitive=false)
```

== Differentiating linear algebra operations <differentiating-linear-algebra-operations>

#table(
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
  )
Credit: Kaiwen Jin for making the table.

== The problem of reverse mode AD

- $O(n)$ overhead in memory - the memory wall problem.

= Optimal checkpointing
== Checkpointing <sec-checkpointing>
#figure(canvas(length: 2cm, {
  import draw: *
  let s(it) = text(16pt, it)
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
    content((i * dx, -0.2), text(16pt)[$s_#i$])
  }
  content((1.25, -0.6), s[(c)])
}),
) <fig:fig-checkpointing>

The optimal strategy for the multiple-pass checkpointing is also known as the _Treeverse algorithm_@Griewank1992.
#figure(canvas(length: 1.5cm, {
  import draw: *
  let s(it) = text(16pt, it)
  let dx = 0.5
  let states(loc, seq, prefix) = {
    for (i, s) in seq.enumerate(){
      circle((i * dx + loc.at(0), loc.at(1)), radius: 0.1, fill: (if (s == 1) {black} else {if (s == 0) {white} else {gray.lighten(20%)}}), name: prefix + str(i))
    }
  }
  states((0, 0), (1, 1, 0, 0, 2, 0), "s")
  content((0.5, -0.7), text(16pt)[new], name: "new")
  line("new", "s1.south", mark: (end: "straight"))
  content((2, -0.2), text(16pt, red)[$times$])
  bezier("s1", "s2.north", (1.5 * dx, 0.8), mark: (end: "straight"))
  bezier("s0", "s3.north", (0.5 * dx, 1.4), mark: (end: "straight"))
  bezier("s4.north", "s5.north", (4.5 * dx, 0.8), mark: (end: "straight"))
  circle((2.25, 0), radius: 0.55, stroke: (dash: "dashed"))
  content((4, 0.5), s[previous pass])
})
)

==
Let us denote the maximum number of steps in the program given by the Treeverse algorithm as $eta(tau, delta)$. Then we have the following recurrence relation:
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k). $ <eq:treeverse-recurrence>
This is based on the following observations:
- In the next sweep, the number of sweeps allowed is decreased by 1, explaining $tau - 1$ on the right side.
- The current sweep devides the program into $delta + 1$ sectors with $delta$ checkpoints. In the next sweep, the number of allowed checkpoints for the $k$-th sector is $delta - k$, which is consistent with number of checkpoints behind the $k$-th sector.

== Binomial checkpointing
What is the function that satisfies the recurrence relation in the above equation?
$ eta(tau, delta) = ((tau + delta)!)/(tau!delta!). $

== Example: 30 steps, 5 checkpoints
#figure(clip(rotate(-90deg, canvas(length: 0.5cm, {
  visualize-treeverse("treeverse-30-5.json")
})), top: 3.8cm, bottom: 3.8cm, left: -4.8cm, right: -4.8cm
))
- $x$ axis, the time steps
- $y$ axis, the computational steps

==
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
)

= Hands-on: Sensitivity Analysis
== Lorenz Equations
The Lorenz system@Lorenz1963 is a classic model for studying chaos, describing dynamics defined in three-dimensional space
$ 
(d x)/(d t) &= sigma(y - x),\
(d y)/(d t) &= x(rho -z) - y,\
(d z)/(d t) &= x y - beta z.
$
where $sigma$, $rho$, and $beta$ are three control parameters.

==

#figure(image("images/lorenz.gif", alt: "Lorenz"))

==

#figure(image("images/lorenz.png", width: 500pt, alt: "Lorenz"))

- left: the gradient
- right: the chaotic and non-chaotic Lorenz curve

== Hands on

- To run, just open the demo folder in a terminal and type:
```bash
$ make init-ADSeismic
$ make example-ADSeismic
```

- Tasks:
  - Modify the number of checkpoints to see the effect. What is the minimum number of checkpoints?
  - Switch the AD engine to ForwardDiff.jl and compare the performance.

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
cases(
  u^(n+1)_(i,j) approx &(Delta t^2)/(1+(zeta_(1i)+zeta_(2j))Delta t/2) (
    (2-zeta_(1i)zeta_(2j))u_(i,j)^n - (1-(zeta_(1i)+zeta_(2j))Delta t/2)/(Delta t^2)u^(n-1)_(i,j) + c_(i,j)^2(u_(i+1,j)^n-2u_(i,j)^n+u_(i-1,j)^n)/(Delta x^2)\ 
    &+ c_(i,j)^2(u_(i,j+1)^n-2u_(i,j)^n+u_(i,j-1)^n)/(Delta y^2)
    + ((phi.alt_x)_(i+1,j)-(phi.alt_x)_(i-1,j))/(2Delta x) + ((phi.alt_y)_(i,j+1)-(phi.alt_y)_(i,j-1))/(2Delta y) ),
  (phi.alt_x)_(i,j)^(n+1) &= (1-Delta t zeta_(1i))(phi.alt_x)_(i,j)^n + Delta t c_(i,j)^2 (zeta_(1i)-zeta_(2j))(u_(i+1,j)-u_(i-1,j))/(2Delta x),
  (phi.alt_y)_(i,j)^(n+1) &= (1-Delta t zeta_(2j))(phi.alt_y)_(i,j)^n + Delta t c_(i,j)^2 (zeta_(2j)-zeta_(1i))(u_(i,j+1)-u_(i,j-1))/(2Delta y)
)
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




== Differentiating linear algebra operations <differentiating-linear-algebra-operations>


== Notations

We derived the following useful relations:
$ tr[A(C compose B)] = sum A^T compose C compose B = tr((C compose A^T)^T B) = tr(C^T compose A)B $ <eq:tr_compose>

$ (C compose A)^T = C^T compose A^T $ <eq:transpose_compose>

Let $cal(L)$ be a real function of a complex variable $x$, $ (partial cal(L))/(partial x^*) = ((partial cal(L))/(partial x))^* $ <eq:diff_complex>



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

== Differentiating implicit functions <differentiating-implicit-functions>

Considering a user-defined mapping $bold(F): RR^d times RR^n -> RR^d$ that encapsulates the optimality criteria of a given problem, an optimal solution, represented as $x(theta)$, is expected to satisfy the root condition of $bold(F)$ as follows:
$ bold(F)(x^*(theta), theta) = 0 $ <eq-Ffunction>

The function $x^*(theta): RR^n -> RR^d$ is implicitly defined. According to the implicit function theorem@Blondel2022, given a point $(x_0, theta_0)$ that satisfies $F(x_0, theta_0) = 0$ with a continuously differentiable function $bold(F)$, if the Jacobian $partial bold(F)/partial x$ evaluated at $(x_0, theta_0)$ forms a square invertible matrix, then there exists a function $x(dot)$ defined in a neighborhood of $theta_0$ such that $x^*(theta_0) = x_0$. Moreover, for all $theta$ in this neighborhood, it holds that $bold(F)(x^*(theta), theta) = 0$ and $(partial x^*)/(partial theta)$ exists. By applying the chain rule, the Jacobian $(partial x^*)/(partial theta)$ satisfies

$ (partial bold(F)(x^*, theta))/(partial x^*) (partial x^*)/(partial theta) + (partial bold(F)(x^*, theta))/(partial theta) = 0 $

Computing $partial x^* / partial theta$ entails solving the system of linear equations expressed as

$ underbrace((partial bold(F)(x^*, theta))/(partial x^*), "V" in RR^(d times d)) underbrace((partial x^*)/(partial theta), "J" in RR^(d times n)) = -underbrace((partial bold(F)(x^*, theta))/(partial theta), "P" in RR^(d times n)) $ <eq-implicit-linear-equation>

Therefore, the desired Jacobian is given by $J = V^(-1)P$. In many practical situations, explicitly constructing the Jacobian matrix is unnecessary. Instead, it suffices to perform left-multiplication or right-multiplication by $V$ and $P$. These operations are known as the vector-Jacobian product (VJP) and the Jacobian-vector product (JVP), respectively. They are valuable for determining $x(theta)$ using reverse-mode and forward-mode automatic differentiation (AD), respectively.

#bibliography("refs.bib")