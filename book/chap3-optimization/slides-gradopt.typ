#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": canvas, draw, tree, vector, decorations, coordinate
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/algorithmic:1.0.3"
#import "@preview/ctheorems:1.1.3": *
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
    title: [Gradient-based Optimization],
    subtitle: [Algorithms and Applications],
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

== Gradient free optimization - Nelder-Mead

The *Nelder-Mead method* (also known as *downhill simplex method* or *polytope method*) is a numerical method used to find a local minimum or maximum of an objective function in a multidimensional space.

=== Video watch: Nelder-Mead method
https://youtu.be/vOYlVvT3W80?si=1wAfjC4Z_p9jgK2T


= Gradient Descent: The Foundation
== Why gradient-based optimization?
Optimization is at the heart of machine learning, engineering design, and scientific computing.

*Key advantages:*
- Leverage gradient (first-order derivative) information
- Fast convergence rates for differentiable functions
- Widely applicable in practice (e.g. machine learning), suited for high-dimensional (a lot of free parameters) problems

== Learning objectives
#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  By the end of this lecture, you will be able to:
  - Understand the mathematical foundations of gradient-based optimization
  - Implement basic and advanced gradient-based algorithms in Julia
  - Choose appropriate optimization methods for different problem types
  - Recognize and troubleshoot common optimization challenges
  - Apply line search techniques to improve convergence
]

== Intuition
#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *Intuition First!*
  
  Imagine you're hiking on a mountain in dense fog. You can't see far ahead, but you can feel the slope under your feet. The gradient tells you:
  - Which direction is steepest uphill (gradient direction)
  - Which direction is steepest downhill (negative gradient)
  - How steep the slope is (magnitude of gradient)
]

== Definition
#definition[
If $f: RR^n arrow RR$ is differentiable, then the vector-valued function $nabla f: RR^n arrow RR^n$ defined by
$ nabla f(bold(x)) = vec(
  (partial f(bold(x)))/(partial x_1),
  (partial f(bold(x)))/(partial x_2),
  dots.v,
  (partial f(bold(x)))/(partial x_n)
) $
is called the gradient of $f$.
]

== Key properties
*Key Properties:*
- The gradient $nabla f(bold(x))$ points in the direction of *steepest ascent*
- The negative gradient $-nabla f(bold(x))$ points in the direction of *steepest descent*
- The magnitude $||nabla f(bold(x))||$ indicates how steep the slope is

*At a local minimum:* $nabla f(bold(x)) = bold(0)$ (the gradient vanishes)!

== The big idea
#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *The Big Idea*
  
  If we want to minimize a function $f(bold(x))$, we should move in the direction that decreases $f$ most rapidly. That direction is $-nabla f(bold(x))$!
]

== Mathematical foundation
Gradient descent is based on the Taylor expansion. Moving $bold(x)$ slightly in the negative gradient direction:
$ f(bold(x) - epsilon nabla f(bold(x))) approx f(bold(x)) - epsilon ||nabla f(bold(x))||_2^2 $

Since $||nabla f(bold(x))||_2^2 >= 0$, we have $f(bold(x) - epsilon nabla f(bold(x))) <= f(bold(x))$ for small $epsilon > 0$.

== The algorithm
The gradient descent update rule:
$ bold(theta)_(t+1) = bold(theta)_t - alpha bold(g)_t $

where:
- $bold(theta)_t$ is our current position (parameter vector) at iteration $t$
- $bold(g)_t = nabla_(bold(theta)_t) f(bold(theta)_t)$ is the gradient at the current position
- $alpha$ is the *learning rate* or step size

== Implementation
```julia
function gradient_descent(f, x; niters::Int, learning_rate::Real)
    history = [x]
    for i = 1:niters
        g = ForwardDiff.gradient(f, x)
        x -= learning_rate * g
        push!(history, x)
    end
    return history
end
```

== Example: The Rosenbrock function
The Rosenbrock function is very popular in optimization research:
$ f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2 $

*Why is this function special?*
- Global minimum at $(1, 1)$ with $f(1, 1) = 0$
- Narrow, curved valley (the "Rosenbrock banana")
- Easy to evaluate but hard to optimize!

== Live coding: Rosenbrock optimization
#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: gradient descent on Rosenbrock function])
}))

== Common problems
#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *What Can Go Wrong?*
  
  1. *Slow convergence* in narrow valleys (like Rosenbrock!)
  2. *Oscillatory behavior* when learning rate is too large
  3. *Getting trapped* in plateaus with tiny gradients
  4. *Wrong direction* near saddle points
]

*Debugging:* If your gradient descent is oscillating, decrease the learning rate!

= Gradient Descent: Advanced Methods
== The momentum analogy
#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *The Momentum Analogy*
  
  Imagine a ball rolling down a hill:
  - *Without momentum*: Stops at first obstacle
  - *With momentum*: Can roll through small bumps and valleys!
  
  Momentum helps us "remember" where we were going and keeps moving in consistent directions.
]

== The mathematics
The momentum method adds a "velocity" term that accumulates gradients over time:

$ bold(v)_(t+1) &= beta bold(v)_t - alpha bold(g)_t quad "(velocity update)" \
bold(theta)_(t+1) &= bold(theta)_t + bold(v)_(t+1) quad "(position update)" $

where:
- $bold(v)_t$ is the *velocity* (momentum) vector  
- $beta$ is the *momentum coefficient* (typically $0.9$)
- $alpha$ is the learning rate

== Implementation
```julia
function gradient_descent_momentum(f, x; niters::Int, β::Real, learning_rate::Real)
    history = [x]
    v = zero(x)  # Initialize velocity to zero
    
    for i = 1:niters
        g = ForwardDiff.gradient(f, x)
        v = β .* v .- learning_rate .* g  # Update velocity
        x += v                           # Move by velocity
        push!(history, x)
    end
    return history
end
```

== Trade-offs
*Advantages:*
- Faster convergence in consistent directions
- Can escape shallow local minima
- Dampens oscillations in ravines

*Disadvantages:*
- Can overshoot near the optimum
- Adds hyperparameter ($beta$) to tune
- May oscillate around the minimum

*Deeper reason:* If we scale the function $f(bold(x))$ by a constant $c > 0$, then the gradient $nabla f(bold(x))$ is scaled by $c$. Therefore, the momentum term $beta bold(v)_t$ is also scaled by $c$. This means that the previous methods are are sensitivy to the scale of the function.

== AdaGrad
#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *The Core Insight*
  
  Different parameters might need different learning rates! 
  - Parameters with *large gradients* → use *smaller* steps
  - Parameters with *small gradients* → use *larger* steps
  
  AdaGrad automatically adapts the learning rate for each parameter.
]

== The algorithm
AdaGrad keeps track of the sum of squared gradients and uses it to scale the learning rate:

$ bold(r)_t &= bold(r)_(t-1) + bold(g)_t^2 quad "(accumulate squared gradients)" \
bold(eta)_t &= frac(alpha, sqrt(bold(r)_t + epsilon)) quad "(adaptive learning rate)" \
bold(theta)_(t+1) &= bold(theta)_t - bold(eta)_t circle.small bold(g)_t quad "(parameter update)" $

*Key components:*
- $bold(r)_t$: accumulated squared gradients (never decreases!)
- $epsilon$: tiny number ($10^(-8)$) to avoid division by zero
- $circle.small$: element-wise multiplication

== Implementation
```julia
function adagrad_optimize(f, x; niters, learning_rate, ϵ=1e-8)
    rt = zero(x)
    η = zero(x)
    history = [x]
    for step in 1:niters
        Δ = ForwardDiff.gradient(f, x)
        @. rt = rt + Δ^2
        @. η = learning_rate / sqrt(rt + ϵ)
        x = x .- Δ .* η
        push!(history, x)
    end
    return history
end
```

== Characteristics
AdaGrad's main advantage is that it automatically adjusts the learning rate and is less sensitive to the choice of initial learning rate. However, the accumulated squared gradients grow monotonically, causing the effective learning rate to shrink over time, potentially leading to *premature convergence*.

== Adam
#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *Why Adam is Popular*
  
  Adam = Momentum + AdaGrad + Bias Correction
  
  It's like having the best of both worlds:
  - *Momentum*: remembers where we're going  
  - *Adaptive rates*: different speeds for different parameters
  - *Bias correction*: fixes startup problems
]

== The complete algorithm
Adam maintains two exponential moving averages:

$ bold(m)_t &= beta_1 bold(m)_(t-1) + (1 - beta_1) bold(g)_t quad "(momentum-like)" \
bold(v)_t &= beta_2 bold(v)_(t-1) + (1 - beta_2) bold(g)_t^2 quad "(AdaGrad-like)" $

Then applies bias correction (important for early iterations):

$ hat(bold(m))_t &= frac(bold(m)_t, 1 - beta_1^t) quad "(corrected first moment)" \
hat(bold(v))_t &= frac(bold(v)_t, 1 - beta_2^t) quad "(corrected second moment)" $

Finally updates parameters:

$ bold(theta)_(t+1) = bold(theta)_t - frac(alpha hat(bold(m))_t, sqrt(hat(bold(v))_t) + epsilon) $

== Default hyperparameters
*Default hyperparameters* (work well in practice):
- $beta_1 = 0.9$ (momentum decay)
- $beta_2 = 0.999$ (gradient variance decay)  
- $alpha = 0.001$ (learning rate)
- $epsilon = 10^(-8)$ (numerical stability)

== Implementation
#text(16pt)[```julia
function adam_optimize(f, x; niters, learning_rate, β1=0.9, β2=0.999, ϵ=1e-8)
    mt = zero(x)
    vt = zero(x)
    βp1 = β1
    βp2 = β2
    history = [x]
    for step in 1:niters
        Δ = ForwardDiff.gradient(f, x)
        @. mt = β1 * mt + (1 - β1) * Δ
        @. vt = β2 * vt + (1 - β2) * Δ^2
        @. Δ = mt / (1 - βp1) / (√(vt / (1 - βp2)) + ϵ) * learning_rate
        βp1, βp2 = βp1 * β1, βp2 * β2
        x = x .- Δ
        push!(history, x)
    end
    return history
end
```]

== Why Adam works
Adam has become one of the most popular optimization algorithms in machine learning due to:
1. Fast convergence in practice
2. Robustness to hyperparameter choices
3. Good performance across a wide range of problems
4. Built-in bias correction for the moment estimates

== The Optimisers.jl Package
Julia provides a comprehensive collection of optimization algorithms through the `Optimisers.jl` package. This package includes implementations of various gradient-based optimizers with a unified interface.

== Available optimizers
#text(14pt)[```julia
using Optimisers, ForwardDiff

# Available optimizers include:
# Descent, Momentum, Nesterov, RMSProp, Adam, AdaGrad, etc.

function optimize_with_optimisers(f, x0, optimizer_type; niters=1000)
    method = optimizer_type(0.01)  # learning rate
    state = Optimisers.setup(method, x0)
    history = [x0]
    x = copy(x0)
    
    for i = 1:niters
        grad = ForwardDiff.gradient(f, x)
        state, x = Optimisers.update(state, x, grad)
        push!(history, copy(x))
    end
    
    return history
end
```]

== Features
The package provides a clean, composable interface where different optimizers can be easily swapped and combined. It also supports advanced features like gradient clipping, weight decay, and learning rate schedules.

= Hessian-based Optimization
== Newton's method
Newton's method is a second-order optimization algorithm that uses both first and second derivatives of the objective function.

The Newton method update rules are:
$ bold(H)_k bold(p)_k &= -bold(g)_k \
bold(x)_(k+1) &= bold(x)_k + bold(p)_k $

where:
- $bold(H)_k$ is the Hessian matrix (matrix of second derivatives) at iteration $k$
- $bold(g)_k$ is the gradient vector at iteration $k$
- $bold(p)_k$ is the Newton step direction

== Newton's method: implementation
```julia
function newton_optimizer(f, x; tol=1e-5)
    k = 0
    history = [x]
    while k < 1000
        k += 1
        gk = ForwardDiff.gradient(f, x)
        hk = ForwardDiff.hessian(f, x)
        dx = -hk \ gk  # Solve Hk * dx = -gk
        x += dx
        push!(history, x)
        sum(abs2, dx) < tol && break
    end
    return history
end
```

== Newton's method: trade-offs
*Advantages:*
- Quadratic convergence near the optimum (very fast)
- Theoretically optimal for quadratic functions
- Scale-invariant (unaffected by linear transformations)

*Disadvantages:*
- Requires computation and inversion of the Hessian matrix: $O(n^3)$ cost per iteration
- Hessian computation requires $O(n)$ times more resources than gradient computation
- May not converge if the Hessian is not positive definite
- Expensive for high-dimensional problems

== The BFGS algorithm
*Video watch: BFGS algorithm*: https://youtu.be/VIoWzHlz7k8

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton method that approximates the Hessian matrix using only gradient information.

The BFGS algorithm updates an approximation $bold(B)_k$ of the Hessian matrix using the secant equation:
$ bold(B)_k bold(p)_k &= -bold(g)_k quad "Newton-like update rule" \
alpha_k &= arg min_alpha f(bold(x) + alpha bold(p)_k) quad "Line search" \
bold(s)_k &= alpha_k bold(p)_k \
bold(x)_(k+1) &= bold(x)_k + bold(s)_k \
bold(y)_k &= bold(g)_(k+1) - bold(g)_k $

== BFGS: implementation
```julia
using Optim, ForwardDiff

function optimize_bfgs(f, x0; iterations=1000)
    options = Optim.Options(iterations=iterations, store_trace=true, extended_trace=true)
    result = optimize(f, x -> ForwardDiff.gradient(f, x), x0, BFGS(), options)
    return result
end
```

== BFGS: key features
*Key Features of BFGS:*
- Superlinear convergence (faster than first-order methods, slower than Newton)
- Only requires gradient information (no Hessian computation)
- Automatically builds up curvature information
- Usually includes line search for step size selection
- Memory requirement: $O(n^2)$ for storing the Hessian approximation

== Limited-Memory BFGS (L-BFGS)
For large-scale problems, the $O(n^2)$ memory requirement of BFGS becomes prohibitive. L-BFGS addresses this by storing only the last $m$ pairs of vectors $(bold(s)_i, bold(y)_i)$ instead of the full Hessian approximation.

L-BFGS uses a two-loop recursion to compute the search direction implicitly, requiring only $O(m dot.op n)$ memory where $m$ is typically 5-20.

```julia
# L-BFGS is available in Optim.jl
result = optimize(rosenbrock, [-1.0, -1.0], LBFGS())
```

= Line Search Methods
== Golden section search
The golden section search is a technique for finding the minimum of a unimodal function by successively narrowing the range of values. It's particularly useful as a line search method within other optimization algorithms.

The method uses the golden ratio $tau = frac(sqrt(5) - 1, 2) approx 0.618$ to place evaluation points that maintain the same proportional reduction in the search interval.

== Golden section search: implementation
#columns(2)[#text(16pt)[```julia
function golden_section_search(f, a, b; tol=1e-5)
    τ = (√5 - 1) / 2  # Golden ratio conjugate
    x1 = a + (1 - τ) * (b - a)
    x2 = a + τ * (b - a)
    f1, f2 = f(x1), f(x2)
    k = 0
    
    while b - a > tol
        k += 1
        if f1 > f2
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + τ * (b - a)
            f2 = f(x2)
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - τ) * (b - a)
            f1 = f(x1)
        end
    end
    
    return f1 < f2 ? (x1, f1) : (x2, f2)
end
```]]

== Properties of golden section search
*Properties:*
- Guaranteed convergence for unimodal functions
- Optimal reduction ratio (minimizes maximum number of evaluations)
- Only requires function evaluations, no derivatives
- Convergence rate: $O(n^(0.618))$ where $n$ is the number of iterations

== Applications in optimization
Line search methods like golden section search are commonly used within gradient-based algorithms to determine optimal step sizes:

1. *Exact Line Search*: Find $alpha^* = arg min_alpha f(bold(x)_k + alpha bold(p)_k)$
2. *Inexact Line Search*: Use conditions like Armijo or Wolfe conditions
3. *Backtracking*: Start with large step and reduce until sufficient decrease

These techniques ensure that each iteration makes substantial progress toward the minimum while maintaining algorithm stability.

= Summary
== Key takeaways
#box(fill: rgb("e6f3ff"), inset: 1.2em, radius: 8pt, width: 100%)[
  *Key Takeaways*
  
  1. *Gradient descent* is the foundation - simple but can be slow
  2. *Momentum* adds memory - helps with consistent directions  
  3. *AdaGrad* adapts learning rates - good for sparse problems
  4. *Adam* combines the best ideas - excellent default choice
  5. *Newton/BFGS* use curvature - fastest for smooth problems
  6. *Line search* optimizes step size - improves any method
  7. *Nelder-Mead* is a gradient free method - useful for low-dimensional problems
]

== Method selection guidelines
*Problem-based Selection:*
- *Small problems (n < 100)*: Newton → BFGS → Adam
- *Medium problems (100 < n < 10k)*: Adam → L-BFGS → BFGS  
- *Large problems (n > 10k)*: Adam → AdaGrad → SGD
- *Non-smooth problems*: Subgradient methods or gradient-free
- *Stochastic/noisy gradients*: Adam → AdaGrad → SGD variants

== Practical advice
*Practical Advice:*
- Start with Adam (good default)
- Monitor convergence carefully  
- Tune learning rate first, other params second
- Use automatic differentiation for exact gradients
- Try multiple random initializations

== Comprehensive comparison
#figure(
  table(
    columns: 6,
    align: center,
    inset: 8pt,
    [*Method*], [*Order*], [*Convergence*], [*Cost per Iter*], [*Memory*], [*Best Use Case*],
    
    [Gradient Descent], [1st], [Linear], [O(n)], [O(n)], [Large-scale, simple],
    [Momentum], [1st], [Linear+], [O(n)], [O(n)], [Consistent gradients],
    [Adam], [1st], [Fast], [O(n)], [O(n)], [Deep learning, general],
    [Newton], [2nd], [Quadratic], [O(n³)], [O(n²)], [Small-scale, accurate],
    [BFGS], [Quasi-2nd], [Superlinear], [O(n²)], [O(n²)], [Medium-scale],
    [L-BFGS], [Quasi-2nd], [Superlinear], [O(mn)], [O(mn)], [Large-scale],
  )
)

== Further reading
- *Ruder (2016)*: "An overview of gradient descent optimization algorithms"
- *Kingma & Ba (2014)*: "Adam: A Method for Stochastic Optimization"  
- *Nocedal & Wright*: "Numerical Optimization" (textbook)
- *Julia packages*: `Optim.jl`, `Optimisers.jl`, `Flux.jl`

*Next lecture preview:* Automatic Differentiation - How to compute gradients efficiently!

