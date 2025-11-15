#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:1.0.3"
#import algorithmic: algorithm

#set math.equation(numbering: "(1)")
#show: thmrules

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em, bottom: 1em), base: none, stroke: black)
#let theorem = thmbox("theorem", "Theorem", base: none, stroke: black)
#let proof = thmproof("proof", "Proof")

#align(center, [= Gradient-based Optimization\
_Jin-Guo Liu_])

#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *Learning Objectives*
  
  By the end of this lecture, you will be able to:
  - Understand the mathematical foundations of gradient-based optimization
  - Implement basic and advanced gradient-based algorithms in Julia
  - Choose appropriate optimization methods for different problem types
  - Recognize and troubleshoot common optimization challenges
  - Apply line search techniques to improve convergence
]

*Why do we need optimization?* 

Optimization is at the heart of machine learning, engineering design, and scientific computing. Whether we're training neural networks, fitting models to data, or designing optimal structures, we need efficient algorithms to find the best solutions.

Gradient-based optimization algorithms leverage the gradient (first-order derivative) of the objective function to iteratively find optimal solutions. These methods are particularly powerful when the objective function is differentiable and can achieve fast convergence rates compared to gradient-free alternatives.

== Understanding the Gradient

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *Intuition First!* 🤔
  
  Imagine you're hiking on a mountain in dense fog. You can't see far ahead, but you can feel the slope under your feet. The gradient tells you:
  - Which direction is steepest uphill (gradient direction)
  - Which direction is steepest downhill (negative gradient)
  - How steep the slope is (magnitude of gradient)
]

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

*Key Properties:*
- The gradient $nabla f(bold(x))$ points in the direction of *steepest ascent*
- The negative gradient $-nabla f(bold(x))$ points in the direction of *steepest descent*
- The magnitude $||nabla f(bold(x))||$ indicates how steep the slope is

*🎯 Interactive Question:* What happens at a local minimum? Think about it before reading on...

*Answer:* At a local minimum, $nabla f(bold(x)) = bold(0)$ (the gradient vanishes)!

== Gradient Descent: The Foundation

#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *The Big Idea* 💡
  
  If we want to minimize a function $f(bold(x))$, we should move in the direction that decreases $f$ most rapidly. That direction is $-nabla f(bold(x))$!
]

=== Mathematical Foundation

Gradient descent is based on the Taylor expansion. Moving $bold(x)$ slightly in the negative gradient direction:
$ f(bold(x) - epsilon nabla f(bold(x))) approx f(bold(x)) - epsilon nabla f(bold(x))^T nabla f(bold(x)) = f(bold(x)) - epsilon ||nabla f(bold(x))||_2^2 $

Since $||nabla f(bold(x))||_2^2 >= 0$, we have $f(bold(x) - epsilon nabla f(bold(x))) <= f(bold(x))$ for small $epsilon > 0$.

=== The Algorithm

The gradient descent update rule is beautifully simple:
$ bold(theta)_(t+1) = bold(theta)_t - alpha bold(g)_t $

where:
- $bold(theta)_t$ is our current position (parameter vector) at iteration $t$
- $bold(g)_t = nabla_(bold(theta)_t) f(bold(theta)_t)$ is the gradient at the current position
- $alpha$ is the *learning rate* or step size

*🎯 Think About It:* What happens if $alpha$ is too large? Too small?

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

=== 🧪 Hands-on Example: The Rosenbrock Function

The Rosenbrock function is the "fruit fly" of optimization research:
$ f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2 $

*Why is this function special?*
- Global minimum at $(1, 1)$ with $f(1, 1) = 0$ ✅
- Narrow, curved valley (the "Rosenbrock banana") 🍌
- Easy to evaluate but hard to optimize!
- Tests algorithm robustness

#box(fill: rgb("fffacd"), inset: 1em, radius: 5pt, width: 100%)[
  *💻 Live Coding Exercise*
  
  Let's implement and test gradient descent together:
]

```julia
# Step 1: Define the Rosenbrock function
function rosenbrock(x)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# Step 2: Compute gradient (we'll use ForwardDiff)
using ForwardDiff
∇f = x -> ForwardDiff.gradient(rosenbrock, x)

# Step 3: Test the gradient at point (0, 0)
println("Gradient at (0,0): ", ∇f([0.0, 0.0]))

# Step 4: Run gradient descent
x0 = [-1.0, -1.0]  # Starting point
history = gradient_descent(rosenbrock, x0; niters=10000, learning_rate=0.002)

println("Final point: ", history[end])
println("Final value: ", rosenbrock(history[end]))
```

*🎯 Experiment Time:* Try different learning rates: 0.001, 0.01, 0.1. What happens?

=== ⚠️ Common Problems with Basic Gradient Descent

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *🚫 What Can Go Wrong?*
  
  1. *Slow convergence* in narrow valleys (like Rosenbrock!)
  2. *Oscillatory behavior* when learning rate is too large
  3. *Getting trapped* in plateaus with tiny gradients
  4. *Wrong direction* near saddle points
]

*🎯 Debugging Exercise:* If your gradient descent is oscillating, what should you do?
- A) Increase learning rate
- B) Decrease learning rate  
- C) Add more iterations
- D) Change starting point

*Answer:* B) Decrease the learning rate!

== Advanced Method 1: Gradient Descent with Momentum

#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *🏃‍♂️ The Momentum Analogy*
  
  Imagine a ball rolling down a hill:
  - *Without momentum*: Stops at first obstacle
  - *With momentum*: Can roll through small bumps and valleys!
  
  Momentum helps us "remember" where we were going and keeps moving in consistent directions.
]

=== The Mathematics

The momentum method adds a "velocity" term that accumulates gradients over time:

$ bold(v)_(t+1) &= beta bold(v)_t - alpha bold(g)_t quad "(velocity update)" \
bold(theta)_(t+1) &= bold(theta)_t + bold(v)_(t+1) quad "(position update)" $

where:
- $bold(v)_t$ is the *velocity* (momentum) vector  
- $beta$ is the *momentum coefficient* (typically $0.9$)
- $alpha$ is the learning rate

=== 💻 Implementation

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

=== ⚖️ Trade-offs

*Advantages:*
- Faster convergence in consistent directions
- Can escape shallow local minima
- Dampens oscillations in ravines

*Disadvantages:*
- Can overshoot near the optimum
- Adds hyperparameter ($beta$) to tune
- May oscillate around the minimum

*🎯 Quick Quiz:* What happens if $beta = 0$? What if $beta = 1$?

== Advanced Method 2: AdaGrad (Adaptive Gradient)

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 The Core Insight*
  
  Different parameters might need different learning rates! 
  - Parameters with *large gradients* → use *smaller* steps
  - Parameters with *small gradients* → use *larger* steps
  
  AdaGrad automatically adapts the learning rate for each parameter.
]

=== The Algorithm

AdaGrad keeps track of the sum of squared gradients and uses it to scale the learning rate:

$ bold(r)_t &= bold(r)_(t-1) + bold(g)_t^2 quad "(accumulate squared gradients)" \
bold(eta)_t &= frac(alpha, sqrt(bold(r)_t + epsilon)) quad "(adaptive learning rate)" \
bold(theta)_(t+1) &= bold(theta)_t - bold(eta)_t circle.small bold(g)_t quad "(parameter update)" $

*Key components:*
- $bold(r)_t$: accumulated squared gradients (never decreases!)
- $epsilon$: tiny number ($10^(-8)$) to avoid division by zero
- $circle.small$: element-wise multiplication

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

AdaGrad's main advantage is that it automatically adjusts the learning rate and is less sensitive to the choice of initial learning rate. However, the accumulated squared gradients grow monotonically, causing the effective learning rate to shrink over time, potentially leading to premature convergence.

== Advanced Method 3: Adam (The Champion!) 🏆

#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *🌟 Why Adam is Popular*
  
  Adam = Momentum + AdaGrad + Bias Correction
  
  It's like having the best of both worlds:
  - *Momentum*: remembers where we're going  
  - *Adaptive rates*: different speeds for different parameters
  - *Bias correction*: fixes startup problems
]

=== The Complete Algorithm

Adam maintains two exponential moving averages:

$ bold(m)_t &= beta_1 bold(m)_(t-1) + (1 - beta_1) bold(g)_t quad "(momentum-like)" \
bold(v)_t &= beta_2 bold(v)_(t-1) + (1 - beta_2) bold(g)_t^2 quad "(AdaGrad-like)" $

Then applies bias correction (important for early iterations):

$ hat(bold(m))_t &= frac(bold(m)_t, 1 - beta_1^t) quad "(corrected first moment)" \
hat(bold(v))_t &= frac(bold(v)_t, 1 - beta_2^t) quad "(corrected second moment)" $

Finally updates parameters:

$ bold(theta)_(t+1) = bold(theta)_t - frac(alpha hat(bold(m))_t, sqrt(hat(bold(v))_t) + epsilon) $

*Default hyperparameters* (work well in practice):
- $beta_1 = 0.9$ (momentum decay)
- $beta_2 = 0.999$ (gradient variance decay)  
- $alpha = 0.001$ (learning rate)
- $epsilon = 10^(-8)$ (numerical stability)

```julia
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
```

Adam has become one of the most popular optimization algorithms in machine learning due to:
1. Fast convergence in practice
2. Robustness to hyperparameter choices
3. Good performance across a wide range of problems
4. Built-in bias correction for the moment estimates

== The Optimisers.jl Package

Julia provides a comprehensive collection of optimization algorithms through the `Optimisers.jl` package. This package includes implementations of various gradient-based optimizers with a unified interface.

```julia
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

# Example usage:
# history = optimize_with_optimisers(rosenbrock, [-1.0, -1.0], Optimisers.Adam)
```

The package provides a clean, composable interface where different optimizers can be easily swapped and combined. It also supports advanced features like gradient clipping, weight decay, and learning rate schedules.

== 📊 Method Comparison & Selection Guide

#box(fill: rgb("f5f5dc"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 The Big Question: Which Optimizer Should I Use?*
  
  *Short Answer:* Start with Adam. It works well for most problems!
  
  *Long Answer:* It depends on your problem...
]

=== Complete Comparison

#figure(
  table(
    columns: 6,
    align: center,
    [*Method*], [*Convergence*], [*Memory*], [*Hyperparameters*], [*Pros*], [*Cons*],
    
    [Gradient Descent], [Linear], [O(n)], [α], [Simple, stable], [Slow, sensitive to α],
    [Momentum], [Superlinear], [O(n)], [α, β], [Faster, stable], [Can overshoot],
    [AdaGrad], [Good], [O(n)], [α], [Auto-adapts], [Learning rate decay],
    [Adam], [Fast], [O(n)], [α, β₁, β₂], [Robust, fast], [More complex],
    [Newton], [Quadratic], [O(n²)], [None], [Very fast], [Expensive O(n³)],
    [BFGS], [Superlinear], [O(n²)], [None], [Fast, no Hessian], [O(n²) memory],
    [L-BFGS], [Superlinear], [O(mn)], [m], [Scalable], [Approximation errors],
  ),
  caption: [Complete comparison of optimization methods (n = problem size, m = memory parameter)]
)

== 🛠️ Practical Guidelines & Tips

=== Method Selection Decision Tree

#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *🌳 Choose Your Optimizer*
  
  1. *Problem size < 100 variables?* → Try Newton or BFGS
  2. *Problem size 100-10,000?* → Use Adam or L-BFGS  
  3. *Problem size > 10,000?* → Start with Adam
  4. *Sparse gradients/NLP?* → Try AdaGrad
  5. *Not sure?* → Use Adam (good default!)
]

=== Hyperparameter Tuning Tips

*Learning Rate Selection:*
- Start with $alpha = 0.001$ for Adam, $alpha = 0.01$ for SGD
- Too large → oscillation/divergence  
- Too small → slow convergence
- Use learning rate schedules: decay over time

*Other Parameters:*
- Adam: Use default $beta_1 = 0.9$, $beta_2 = 0.999$
- Momentum: Try $beta = 0.9$ or $beta = 0.99$
- L-BFGS: Use $m = 10$ (memory parameter)

=== 🚨 Troubleshooting Common Issues

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *Problem:* Loss is exploding/NaN values
  *Solution:* Reduce learning rate, check gradient computation
  
  *Problem:* Very slow convergence  
  *Solution:* Increase learning rate, try Adam instead of SGD
  
  *Problem:* Oscillating around minimum
  *Solution:* Reduce learning rate, add momentum
  
  *Problem:* Stuck in plateau
  *Solution:* Check gradient computation, try different initialization
]

=== Convergence Monitoring

Monitor these metrics during optimization:
1. *Function value* $f(bold(theta)_t)$ (should decrease)
2. *Gradient norm* $||nabla f(bold(theta)_t)||$ (should approach 0)
3. *Parameter change* $||bold(theta)_(t+1) - bold(theta)_t||$ (should get smaller)

Stop when: gradient norm < threshold OR parameter change < threshold OR max iterations reached.

== Hessian-based Optimization

=== Newton's Method

Newton's method is a second-order optimization algorithm that uses both first and second derivatives of the objective function. It approximates the function locally as a quadratic and finds the minimum of this quadratic approximation.

The Newton method update rules are:
$ bold(H)_k bold(p)_k &= -bold(g)_k \
bold(x)_(k+1) &= bold(x)_k + bold(p)_k $

where:
- $bold(H)_k$ is the Hessian matrix (matrix of second derivatives) at iteration $k$
- $bold(g)_k$ is the gradient vector at iteration $k$
- $bold(p)_k$ is the Newton step direction

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

*Advantages of Newton's Method:*
- Quadratic convergence near the optimum (very fast)
- Theoretically optimal for quadratic functions
- Scale-invariant (unaffected by linear transformations)

*Disadvantages:*
- Requires computation and inversion of the Hessian matrix: $O(n^3)$ cost per iteration
- Hessian computation requires $O(n)$ times more resources than gradient computation
- May not converge if the Hessian is not positive definite
- Expensive for high-dimensional problems

=== The BFGS Algorithm

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton method that approximates the Hessian matrix using only gradient information. It belongs to the class of quasi-Newton methods, which build up curvature information over iterations.

The BFGS algorithm updates an approximation $bold(B)_k$ of the Hessian matrix using the secant equation:
$ bold(B)_k bold(p)_k &= -bold(g)_k quad "Newton-like update rule" \
alpha_k &= arg min_alpha f(bold(x) + alpha bold(p)_k) quad "Line search" \
bold(s)_k &= alpha_k bold(p)_k \
bold(x)_(k+1) &= bold(x)_k + bold(s)_k \
bold(y)_k &= bold(g)_(k+1) - bold(g)_k $

The Hessian approximation is updated using the BFGS formula:
$ bold(B)_(k+1) = bold(B)_k + frac(bold(y)_k bold(y)_k^T, bold(y)_k^T bold(s)_k) - frac(bold(B)_k bold(s)_k bold(s)_k^T bold(B)_k^T, bold(s)_k^T bold(B)_k bold(s)_k) $

This update satisfies the secant equation: $bold(B)_(k+1) bold(s)_k = bold(y)_k$.

```julia
using Optim, ForwardDiff

function optimize_bfgs(f, x0; iterations=1000)
    options = Optim.Options(iterations=iterations, store_trace=true, extended_trace=true)
    result = optimize(f, x -> ForwardDiff.gradient(f, x), x0, BFGS(), options)
    return result
end

# Example usage
result = optimize_bfgs(rosenbrock, [-1.0, -1.0])
```

*Key Features of BFGS:*
- Superlinear convergence (faster than first-order methods, slower than Newton)
- Only requires gradient information (no Hessian computation)
- Automatically builds up curvature information
- Usually includes line search for step size selection
- Memory requirement: $O(n^2)$ for storing the Hessian approximation

=== Limited-Memory BFGS (L-BFGS)

For large-scale problems, the $O(n^2)$ memory requirement of BFGS becomes prohibitive. L-BFGS addresses this by storing only the last $m$ pairs of vectors $(bold(s)_i, bold(y)_i)$ instead of the full Hessian approximation.

L-BFGS uses a two-loop recursion to compute the search direction implicitly, requiring only $O(m dot.op n)$ memory where $m$ is typically 5-20.

```julia
# L-BFGS is available in Optim.jl
result = optimize(rosenbrock, [-1.0, -1.0], LBFGS())
```

== Line Search Methods

=== Golden Section Search

The golden section search is a technique for finding the minimum of a unimodal function by successively narrowing the range of values. It's particularly useful as a line search method within other optimization algorithms.

The method uses the golden ratio $tau = frac(sqrt(5) - 1, 2) approx 0.618$ to place evaluation points that maintain the same proportional reduction in the search interval.

```julia
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

# Example: Find minimum of (x-4)²
x_min, f_min = golden_section_search(x -> (x-4)^2, -5, 5)
```

*Properties of Golden Section Search:*
- Guaranteed convergence for unimodal functions
- Optimal reduction ratio (minimizes maximum number of evaluations)
- Only requires function evaluations, no derivatives
- Convergence rate: $O(n^(0.618))$ where $n$ is the number of iterations

=== Applications in Optimization

Line search methods like golden section search are commonly used within gradient-based algorithms to determine optimal step sizes:

1. *Exact Line Search*: Find $alpha^* = arg min_alpha f(bold(x)_k + alpha bold(p)_k)$
2. *Inexact Line Search*: Use conditions like Armijo or Wolfe conditions
3. *Backtracking*: Start with large step and reduce until sufficient decrease

These techniques ensure that each iteration makes substantial progress toward the minimum while maintaining algorithm stability.

== Summary and Method Selection

#figure(
  table(
    columns: 6,
    align: center,
    [*Method*], [*Order*], [*Convergence*], [*Cost per Iter*], [*Memory*], [*Best Use Case*],
    
    [Gradient Descent], [1st], [Linear], [O(n)], [O(n)], [Large-scale, simple],
    [Momentum], [1st], [Linear+], [O(n)], [O(n)], [Consistent gradients],
    [Adam], [1st], [Fast], [O(n)], [O(n)], [Deep learning, general],
    [Newton], [2nd], [Quadratic], [O(n³)], [O(n²)], [Small-scale, accurate],
    [BFGS], [Quasi-2nd], [Superlinear], [O(n²)], [O(n²)], [Medium-scale],
    [L-BFGS], [Quasi-2nd], [Superlinear], [O(mn)], [O(mn)], [Large-scale],
  ),
  caption: [Comprehensive comparison of optimization methods. Here $n$ is the problem dimension and $m$ is the L-BFGS memory parameter.]
)

== 🎓 Lecture Summary

#box(fill: rgb("e6f3ff"), inset: 1.2em, radius: 8pt, width: 100%)[
  *🌟 Key Takeaways*
  
  1. *Gradient descent* is the foundation - simple but can be slow
  2. *Momentum* adds memory - helps with consistent directions  
  3. *AdaGrad* adapts learning rates - good for sparse problems
  4. *Adam* combines the best ideas - excellent default choice
  5. *Newton/BFGS* use curvature - fastest for smooth problems
  6. *Line search* optimizes step size - improves any method
]

=== Method Selection Guidelines

*Problem-based Selection:*
- *Small problems (n < 100)*: Newton → BFGS → Adam
- *Medium problems (100 < n < 10k)*: Adam → L-BFGS → BFGS  
- *Large problems (n > 10k)*: Adam → AdaGrad → SGD
- *Non-smooth problems*: Subgradient methods or gradient-free
- *Stochastic/noisy gradients*: Adam → AdaGrad → SGD variants

*🎯 Practical Advice:*
- Start with Adam (good default)
- Monitor convergence carefully  
- Tune learning rate first, other params second
- Use automatic differentiation for exact gradients
- Try multiple random initializations

== 🏋️‍♂️ Exercises & Next Steps

=== Hands-on Exercises

#box(fill: rgb("fffacd"), inset: 1em, radius: 5pt, width: 100%)[
  *💻 Programming Exercises*
  
  1. *Compare optimizers* on the Rosenbrock function:
     - Implement SGD, Momentum, Adam from scratch
     - Plot convergence curves for different learning rates
     - Which method reaches the minimum fastest?
  
  2. *Hyperparameter sensitivity*:
     - Test Adam with different β₁, β₂ values  
     - How does performance change?
     - Find the best hyperparameters for your problem
  
  3. *Real-world application*:
     - Fit a neural network to classify handwritten digits
     - Compare SGD vs Adam training curves
     - Which converges faster? Which generalizes better?
]

=== 🔍 Further Reading

- *Ruder (2016)*: "An overview of gradient descent optimization algorithms"
- *Kingma & Ba (2014)*: "Adam: A Method for Stochastic Optimization"  
- *Nocedal & Wright*: "Numerical Optimization" (textbook)
- *Julia packages*: `Optim.jl`, `Optimisers.jl`, `Flux.jl`

*Next lecture preview:* Automatic Differentiation - How to compute gradients efficiently! 🚀
