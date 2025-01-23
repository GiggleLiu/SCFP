#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#import "@preview/ctheorems:1.1.3": *

#show: book-page.with(title: "Polynomial optimization")

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)

= Integer programming

_Linear Programming_ deals with the problem of optimizing a *linear objective function* subject to *linear equality and inequality constraints* on *real* variables. It is a convex optimization problem, hence it is easy to solve.

_Integer Programming_ is similar, but deals with *integer* variables. It is a non-convex optimization problem, and belongs to the complexity class of NP-complete.
For convex optimization problems, both the feasible region and the objective function are convex. However, for integer programming problems, the feasible region $bb(Z)$ is not convex.

== Example 1: Branching and cut for solving integer programming
Let us consider the following integer programming problem (source: #link("https://youtu.be/upcsrgqdeNQ?si=B5uilXqSrI5Jg976", "Youtube")):
$
  z = 5x_1 + 6x_2\
  x_1 + x_2 <= 5\
  4x_1 + 7x_2 <= 28\
  x_1, x_2 >= 0\
  #box(stroke: black, inset: 5pt, [$x_1, x_2 in bb(Z)$])
$

#figure(canvas(length:0.9cm, {
  import plot
  import draw: *
  plot.plot(size: (10,7),
    x-tick-step: 1,
    y-tick-step: 1,
    x-label: [$x_1$],
    y-label: [$x_2$],
    y-max: 5,
    y-min: 0,
    x-max: 5,
    x-min: 0,
    name: "plot",
    {
      let f1 = x => 5 - x
      let f2 = x => (28 - 4 * x) / 7
      let f3 = x => (23 - 5 * x) / 6
      plot.add-fill-between(
        domain: (0, 5),
        x => calc.min(f1(x), f2(x)),
        x => 0,
        label: [feasible region],
        style: (fill: green.lighten(70%))
      )
      plot.add(
        domain: (0, 5),
        f1,
        label: [$x_1 + x_2 = 5$],
        style: (stroke: blue)
      )

      plot.add(
        domain: (0, 5),
        f2,
        label: [$4x_1 + 7x_2 = 28$],
        style: (stroke: orange)
      )

      plot.add(
        domain: (0, 5),
        f3,
        label: [objective function],
        style: (stroke: (paint: black, dash: "dashed")),
      )
 
      plot.add(range(36).map(x => (calc.rem(x, 6), calc.div-euclid(x, 6))), style: (stroke: none), mark: "o", mark-style: (fill: none), mark-size: 0.1)
      let a = 2
      plot.add-anchor("obj", (a, f3(a)))
      plot.add-anchor("p1", (2.33, 2.67))
      plot.add-anchor("p2", (3, 2))
      plot.add-anchor("p3", (1.75, 3))
      plot.add-anchor("p5", (1, 3.42))
      plot.add-anchor("p6", (0, 4))
      plot.add-anchor("p7", (1, 3))
    }
  )
  line("plot.obj", (rel: (0.4, 0.6), to: "plot.obj"), mark: (end: "straight"))
  circle("plot.p1", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p1"), [$p_1$])
  circle("plot.p2", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p2"), [$p_2$])
  circle("plot.p3", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p3"), [$p_3$])
  circle("plot.p5", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p5"), [$p_5$])
  circle("plot.p6", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p6"), [$p_6$])
  circle("plot.p7", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p7"), [$p_7$])
}),
caption: [Integer programming problem. The fesible region of linear programming problem is the green area, while the feasible region of integer programming problem are the discrete points marked by small circles.],
) <fig:integer-programming>

We first solve the linear programming problem by relaxing the integer constraint, i.e. removing $x_1, x_2 in bb(Z)$ constraint. The feasible region is the green polygon in @fig:integer-programming, and the optimal solution is the point $p_1$ at the line crossing.

$p_1$ is not feasible for the integer programming problem due to $x_2$ being non-integer. To force the variables to be integer, we add some inequalities constraints to the linear programming problem as shown in @fig:branching-and-cutting. Since $x_2$ is non-integer, we create two sub-problems (or branches) by adding two inequalities constraints $x_2 <= 2$ and $x_2 >= 3$ to the linear programming problem.
It turns out the sub-problem 2 with $x_2 <= 2$ accepts integer solution $p_2$ as the optimal solution. So we stop this branch and continue to solve the sub-problem 3 with $x_2 >= 3$.

#figure(canvas(length: 0.85cm, {
  import draw: *
  let boxed(c) = box(stroke: black, inset: (x: 7pt, y: 5pt), c)
  content((0, 0), boxed(text(10pt)[Sub-problem 1
  - $z = 27.67$
  - $x_1 = 2.33$
  - $x_2 = 2.67$]), name: "sub-problem-1")
  content((-4, -3), boxed(text(10pt)[Sub-problem 2
  - $z = 27$
  - $x_1 = 3$
  - $x_2 = 2$]), name: "sub-problem-2")
  content((4, -3), boxed(text(10pt)[Sub-problem 3
  - $z = 26.75$
  - $x_1 = 1.75$
  - $x_2 = 3$]), name: "sub-problem-3")
  content((1, -6), boxed(text(10pt)[Sub-problem 4
  - infeasible]), name: "sub-problem-4")

  content((7, -6.5), boxed(text(10pt)[Sub-problem 5
  - $z = 25.57$
  - $x_1 = 1$
  - $x_2 = 3.42$]), name: "sub-problem-5")

  content((4, -10), boxed(text(10pt)[Sub-problem 6
  - $z = 24$
  - $x_1 = 0$
  - $x_2 = 4$]), name: "sub-problem-6")

  content((10, -10), boxed(text(10pt)[Sub-problem 7
  - $z = 23$\
  - $x_1 = 1$\
  - $x_2 = 3$]), name: "sub-problem-7")

  line("sub-problem-1", "sub-problem-2.north", mark: (end: "straight"), name: "l12")
  content((rel: (-1, 0.1), to: "l12.mid"), text(10pt)[$x_2 <= 2$])
  line("sub-problem-1", "sub-problem-3.north", mark: (end: "straight"), name: "l13")
  content((rel: (1, 0.1), to: "l13.mid"), text(10pt)[$x_2 >= 3$])
  line("sub-problem-3", "sub-problem-4", mark: (end: "straight"), name: "l34")
  content((rel: (-0.8, 0.1), to: "l34.mid"), text(10pt)[$x_1 >= 2$])
  line("sub-problem-3", "sub-problem-5", mark: (end: "straight"), name: "l35")
  content((rel: (0.8, 0.1), to: "l35.mid"), text(10pt)[$x_1 <= 1$])
  line("sub-problem-5", "sub-problem-6", mark: (end: "straight"), name: "l56")
  content((rel: (-0.8, 0.1), to: "l56.mid"), text(10pt)[$x_2 >= 4$])
  line("sub-problem-5", "sub-problem-7", mark: (end: "straight"), name: "l57")
  content((rel: (0.8, 0.1), to: "l57.mid"), text(10pt)[$x_2 <= 3$])
}),
caption: [Branching and cutting for solving integer programming. The additional constraints are marked along the lines. The optimal solution is annotated in each sub-problem.],
) <fig:branching-and-cutting>

Finally, we compare all sub-problems producing integer solutions, and find the optimal solution is $p_2$.

In Julia programming language, we can solve the linear/integer programming problem by using #link("https://github.com/jump-dev/JuMP.jl", "JuMP") package.

```julia
# Sub-problem 1
using JuMP, HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4 * x1 + 7 * x2 <= 28)

# Sub-problem 2: uncomment the following line
# @constraint(model, x2 <= 2)

# Sub-problem 3: uncomment the following line
# @constraint(model, x2 >= 3)

# Sub-problem 4: uncomment the following 2 lines
# @constraint(model, x2 >= 3)
# @constraint(model, x1 >= 2)

# Sub-problem 5: uncomment the following 2 lines
# @constraint(model, x2 <= 2)
# @constraint(model, x1 <= 1)

# Sub-problem 6: uncomment the following 2 lines
# @constraint(model, x1 <= 1)
# @constraint(model, x2 >= 4)

# Sub-problem 7: uncomment the following 2 lines
# @constraint(model, x1 <= 1)
# @constraint(model, x2 <= 3)

@objective(model, Max, 5 * x1 + 6 * x2)

optimize!(model)
value(x1), value(x2), objective_value(model)
```
You can see the result of the linear programming problem.
```output
(2.3333333333333335, 2.6666666666666665, 27.666666666666668)
```
By commenting the lines in the sub-problem, we can solve different sub-problems.
In practice, you do not need to write the branch and cut by yourself.
The solvers in `JuMP` package will do it for you.

```julia
using JuMP, HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, x1 >= 0, Int)
@variable(model, x2 >= 0, Int)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4 * x1 + 7 * x2 <= 28)

@objective(model, Max, 5 * x1 + 6 * x2)

optimize!(model)
value(x1), value(x2), objective_value(model)
```
You can see the result of the integer programming problem.
```output
(3.0, 2.0, 27.0)
```

Note here, the `HiGHS` solver backend is used. Please refer to the #link("https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers")[documentation] to check alternative solvers supporting integer programming.

== Example 2: Spin glass problem
The spin glass energy function is defined as
$
E(sigma) = sum_(i < j) J_(i j) sigma_i sigma_j + sum_i h_i sigma_i
$
where $sigma_i = plus.minus 1$ is the spin of the $i$-th particle, $J_(i j)$ is the interaction strength between the $i$-th and $j$-th particles, and $h_i$ is the external magnetic field.

Finding a spin configuration with the minimum energy is a famous NP-complete problem.
At a first look, it is not an integer programming problem because there is a quadratic term in the energy function.
However, it turns out that the problem can be reduced to an integer programming problem by introducing a auxiliary variable $d_(i j) = sigma_i sigma_j$.

```julia
using JuMP, HiGHS, LinearAlgebra

function solve_spin_glass(J::Matrix, h::Vector; verbose = false)
    n = length(h)
    @assert size(J, 1) == size(J, 2) == n "The size of J and h must be the same!"
    @assert ishermitian(J) "J must be a Hermitian matrix!"

    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= s[i = 1:n] <= 1, Int)            # spin: 0 -> 1, 1 -> -1
    @variable(model, 0 <= d[i = 1:n, j = i+1:n] <= 1, Int) # d[i,j] = s_i s_j

    for i = 1:n
        for j = i+1:n
            # map: (0, 0), (1, 1) -> 0 and (0, 1), (1, 0) -> 1
            @constraint(model, d[i,j] <= s[i] + s[j])
            @constraint(model, d[i,j] <= 2 - s[i] - s[j])
            @constraint(model, d[i,j] >= s[i] - s[j])
            @constraint(model, d[i,j] >= s[i] - s[j])
        end
    end
    
    @objective(model, Min, sum(J[i,j] * (1 - 2 * d[i,j]) for i = 1:n, j = i+1:n) + sum((1 - 2 * s[i]) * h[i] for i = 1:n))
    optimize!(model)
    energy = objective_value(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return energy, 1 .- 2 .* value.(s)
end

J = triu(fill(1.0, 10, 10), 1)
J += J'  # anti-ferromagnetic complete graph
h = zeros(size(J, 1))
Emin, configuration = solve_spin_glass(J, h)
```

The result is
```output
(-5.000000000000282, [1.0, 1.0000000000000648, 0.9999999999999196, -1.0, 0.9999999999999708, 1.0, -1.0, -1.0, -1.0, -1.0])
```

== Example 3: Tropical matrix factorization

Tropical number is defined by replacing the addition and multiplication of real numbers by the maximum and addition of real numbers. i.e. the tropical addition and multiplication are defined as
$
a plus.circle b &= max(a, b)\
a times.circle b &= a + b\
bb(0) &= -infinity\
bb(1) &= 0
$
where $bb(0)$ and $bb(1)$ are the additive identity and multiplicative identity of the tropical numbers.

The tropical matrix multiplication is easy to compute. e.g.
$
mat(1, 2; -infinity, 4) times.circle mat(5, 2; 0, 1) = mat(6, 3; 4, 5)
$

However, the inverse problem turns out to be NP-complete.

#definition([_(Boolean tropical matrix factorization problem)_ Given an $m times n$ tropical matrix $C_(m times n)$ and an integer $k$, are there two Boolean matrices $A_(m times k)$ and $B_(k times n)$ such that $C = A times.circle B$.])

This problem is NP-complete for $k >= 7$@Shitov2014. Even in the special case that all variables are either $-infinity$ or $0$, the problem is not easy. By mapping $-infinity$ to `false` and $0$ to `true`, we have
$
C_(i j) = or.big_(l=1)^k A_(i l) and B_(l j)\
c_(i) = or.big_(l=1)^k A_(i l) and b_(l)
$
if $c_i = 0$, then there exists some $l$ such that $A_(i l) = 0$ and $b_l = 0$.
if $c_i = -infinity$, then there is no such $l$ that $A_(i l) = 0$ and $b_l = 0$.

In practise, we can solve the problem by reducing it to an integer programming problem.
To facilitate the reduction, we introduce a auxiliary tensor $D_(i l j) = A_(i l) and B_(l j)$. The resulting integer programming problem is as follows:

$
&min 1\ 
"s.t." &D_(i l j) <= A_(i l)\ 
&D_(i l j) <= B_(l j) \ 
&D_(i l j) >= A_(i l) + B_(l j) - 1\
&C_(i j) <= sum_(l=1)^k D_(i l j)\ 
&D_(i l j) <= C_(i j)\ 
&A_(i l), B_(l j), D_(i l j) in {0,1}
$

Since we do not have any objective function, we set the 
The lines 2-4 encode the constraint $D_(i l j) = A_(i l) and B_(l j)$ with inequalities. The lines 5-6 encodes the tropical matrix multiplication constraints.

```julia
using JuMP, HiGHS, TropicalNumbers

function tropical_factorization(C::Matrix{TropicalAndOr}, k::Int; verbose = false)
    # C = A * B
    # A: m x k, B: k x n

    m,n = size(C)
    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= a[i = 1:m*k] <= 1, Int) # a[i,l] = a[(l-1)*m+i]
    @variable(model, 0 <= b[i = 1:k*n] <= 1, Int)   # b[l,j] = b[(j-1)*k+l]
    @variable(model, 0 <= d[i = 1:m*k*n] <= 1, Int) # d[i,l,j] = d[(j-1)*k*m+(l-1)*m+i]
    
    for i in 1:m
        for j in 1:n
            if C[i,j].n
                for l in 1:k
                    @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= a[(l-1)*m+i])
                    @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= b[(j-1)*k+l])
                end
                @constraint(model, sum(d[(j-1)*k*m+(l-1)*m+i] for l in 1:k) >= 1)
            else
                for l in 1:k
                    @constraint(model, d[(j-1)*k*m+(l-1)*m+i] + 1 >= a[(l-1)*m+i] + b[(j-1)*k+l])
                    @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= 0)
                end
            end
        end
    end
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return reshape([TropicalAndOr(v ≈ 1.0) for v in value.(a)],m,k), reshape([TropicalAndOr(v ≈ 1.0) for v in value.(b)],k,n)
end

A, B = TropicalAndOr.(rand(Bool, 10, 5)), TropicalAndOr.(rand(Bool, 5, 10))
C = A * B

A2, B2 = tropical_factorization(C, 5)
A2 * B2 == C
```

#bibliography("refs.bib")