#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
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

// == A brief history of autodiff

// - 1964 (*forward mode AD*) ~ Robert Edwin Wengert, A simple automatic derivative evaluation program.
// - 1970 (*backward mode AD*) ~ Seppo Linnainmaa, Taylor expansion of the accumulated rounding error.
// - 1986 (*AD for machine learning*) ~ Rumelhart, D. E., Hinton, G. E., and Williams, R. J., Learning representations by back-propagating errors.
// - 1992 (*optimal checkpointing*) ~ Andreas Griewank, Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.
// - 2000s ~ The boom of tensor based AD frameworks for machine learning.
// - 2020 (*AD on LLVM*) ~ Moses, William and Churavy, Valentin, Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients.

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

The computing time grows *linearly* as the number of variables that we want to differentiate. But does not grow significantly with the number of outputs.

= Reverse mode AD

On the other side, the back-propagation can differentiate *many inputs* with respect to a *single output* efficiently

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

#figure(
  canvas({
    import draw: *
    content((), [TODO: Treeverse Example.])
  }),
  caption: [(a) The Treeverse algorithm ($tau=3$, $delta=3$) and (b) Bennett's algorithm ($k=2$, $n=4$) solutions to the pebble game, the $x$ direction is the grid layout and the $y$ direction is the number of steps. Here, a black circle with white fill represents a pebble returned to the free pool in current step, a black circle with black fill represents the pebble added to the board in current step, and a gray circle represents pebbles left on the board. In (a), red grids are painted grids. In (b), the grid with a flag sign is the goal.]
)

#algorithm({
  import algorithmic: *
  Function("Treeverse", args: ([$S$], [$overline(s_phi.alt)$], [$delta$], [$tau$], [$beta$], [$sigma$], [$phi.alt$]), {
    If(cond: [$sigma > beta$], {
      Assign([$delta$], [$delta - 1$])
      Assign([$s$], [$S[beta] quad$ #Ic([Load initial state $s_beta$])])
      For(cond: [$j = beta, beta+1, dots, sigma-1$], {
        Assign([$s_(j+1)$], [$f_j(s_j) quad$ #Ic([Compute $s_sigma$])])
      })
      Assign([$S[sigma]$], [$s_sigma$])
    })
    Cmt[Recursively call Treeverse with optimal split point $kappa$ (binomial distribution)]
    While(cond: [$tau > 0$ and $kappa = "mid"(delta, tau, sigma, phi.alt) < phi.alt$], {
      Assign([$overline(s_kappa)$], [treeverse($S$, $overline(s_phi.alt)$, $delta$, $tau$, $sigma$, $kappa$, $phi.alt$)])
      Assign([$tau$], [$tau - 1$])
      Assign([$phi.alt$], [$kappa$])
    })
    Assign([$overline(s_sigma)$], [$overline(f_sigma)(overline(s_(sigma+1)), s_sigma) quad$ #Ic([Use existing $s_sigma$ and $overline(s_phi.alt)$ to return gradient])])
    
    If(cond: [$sigma > beta$], {
      State[remove($S[sigma]$)  $quad$ #Ic([Remove $s_sigma$ from cached state set])]
    })
    Return[$overline(s_sigma)$]
  })

  Function("mid", args: ([$delta$], [$tau$], [$sigma$], [$phi.alt$]), {
    Cmt[Select the binomial distribution split point]
    Assign([$kappa$], [$ceil((delta sigma + tau phi.alt)/(tau+delta))$])
    If(cond: [$kappa >= phi.alt$ and $delta > 0$], {
      Assign([$kappa$], [max($sigma+1$, $phi.alt-1$)])
    })
    Return[$kappa$]
  })
})

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

#bibliography("refs.bib")
