#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "@preview/cetz:0.2.2": *
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

= Finite difference
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

= Forward mode AD

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

= Reverse mode AD

On the other side, the back-propagation can differentiate **many inputs** with respect to a **single output** efficiently

$
(partial y_(i+1))/(partial y_i) = (partial y_(i+1))/(partial y_i)
$

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
  content((12, -2.5), s[`...`], name: "...")
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

= Obtaining Hessian

The second order gradient, Hessian, is also recognized as the Jacobian of the gradient. In practice, we can compute the Hessian by differentiating the gradient function with forward mode AD, which is also known as the forward-over-reverse mode AD.

= Optimal checkpointing
The main drawback of the reverse mode AD is the memory usage. The memory usage of the reverse mode AD is proportional to the number of intermediate variables, which scales linearly with the number of operations. The optimal checkpointing@Griewank2008 is a technique to reduce the memory usage of the reverse mode AD. It is a trade-off between the memory and the computational cost. The optimal checkpointing is a step towards solving the memory wall problem

Given the binomial function $eta(tau, delta) = ((tau + delta)!)/(tau!delta!)$, show that the following statement is true.
$ eta(tau,delta) = sum_(k=0)^delta eta(tau-1,k) $

#bibliography("refs.bib")
