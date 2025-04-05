#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations, coordinate
#import "@preview/algorithmic:0.1.0"
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

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Mathematical Optimization],
  subtitle: [Linear Programming and Integer Programming],
  author: [Jin-Guo Liu],
  date: datetime.today(),
  institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
)

// Extract methods
#let (init, slides) = utils.methods(m)
#show: init

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(m)
#show: slides.with()

#outline-slide()

== Convex optimization
Convex: $f(a x + (1-a) y) <= a f(x) + (1-a) f(y)$, e.g. a quadratic function $f(x) = x^2$ is convex, but $f(x) = x^3$ is not.

Convex optimization:
- The feasible region is convex.
- The objective function is convex.

= Linear Programming (LP)
_Linear Programming_ deals with the problem of optimizing a *linear objective function* subject to *linear equality and inequality constraints* on *real* variables. It is a convex optimization problem, hence it is easy to solve. A linear programming problem of $n$ variables is formulated as the following _canonical form_:
$
  max_(x) quad &z = c^T x,\
  "s.t." quad &A x <= b,\
  &x >= 0\
  &#box(stroke: black, inset: 5pt, [$x in bb(R)^n$])
$ <eq:linear-programming>
where $c in bb(R)^n$ is the objective function coefficient vector, $A in bb(R)^(m times n)$ is the constraint matrix, and $b in bb(R)^m$ is the constraint vector.
Note here, the positivity constraint $x >= 0$ is not absolutely necessary. Adding this extra constraint does not sacrifice the generality of linear programming, because any linear program can be written into a canonical form by shifting the variables.

== Example: The Diet Problem
In the diet model, a list of available foods is given together with the nutrient content and the cost per unit weight of each food. A certain amount of each nutrient is required per day. For example, here is the data corresponding to two types of food (fish and rice) and three types of nutrients (starch, proteins, vitamins):
#align(center, table(
  columns: (auto, auto, auto, auto, auto),
  table.header([], [Starch], [Proteins], [Vitamins], [Cost (RMB/kg)]),
  [fish], [0], [4], [2], [6],
  [rice], [7], [2], [1], [3.5],
))

Nutrient content and cost per kg of food. The requirement per day of starch, proteins and vitamins is 8, 15 and 3 respectively. The problem is to find how much of each food to consume per day so as to get the required amount per day of each nutrient at minimal cost.

_Solution_:
In the diet problem, a very natural choice of decision variables is:
- $x_1$: number of units of fish to be consumed per day,
- $x_2$: number of units of rice to be consumed per day.

The objective function is the function to be minimized. In this case, the objective is to minimize the total cost per day which is given by $z = 6x_1 + 3.5x_2$. Finally, we need to describe the different constraints that need to be satisfied by $x_1$ and $x_2$.
This diet problem can therefore be formulated by the following linear program:
$
min_(x_1, x_2) & z = 6x_1 + 3.5x_2\
"s.t." & 0x_1 + 7x_2 ≥ 8\
& 4x_1 + 2x_2 ≥ 15\
& 2x_1 + x_2 ≥ 3\
& x_1 ≥ 0, x_2 ≥ 0.
$

The Julia code is as follows:

#box(text(16pt)[```julia
using JuMP, HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

@constraint(model, 0x1 + 7x2 >= 8)
@constraint(model, 4x1 + 2x2 >= 15)
@constraint(model, 2x1 + x2 >= 3)

@objective(model, Min, 6x1 + 3.5x2)

optimize!(model)

value(x1), value(x2), objective_value(model)
```
])

The result is
#box(text(16pt)[```output
(3.25, 1.0, 23.0)
```])
The minimum cost is 23 RMB per day.


// Its _dual problem_ is
// $
//   min_(y) &w = b^T y,\
//   "s.t." &A^T y >= c,\
//   &y >= 0\
//   &y in bb(R)^m
// $
// and the original problem is the _primal problem_. The dual of the dual is the primal.

// #box(stroke: black, inset: 10pt, width: 100%, [(Weak duality) If $x^*$ is optimal for the primal and $y^*$ is optimal for the dual, then $z^* <= w^*$.])

// #proof([Let $x^*$ be feasible for the primal and $y^*$ be feasible for the dual. Then we have
// $
//   z^* = c^T x^* <= y^(* T)A x^* <= y^(* T)b = w^* arrow.double.r z^* <= w^*
// $])

// What is more surprising is the fact that this inequality is in most cases an equality

// #align(center, box(stroke: 1pt, inset: 10pt, width: 100%, [(Strong duality) If $z^*$ is finite then so is $w^*$ and $z^* = w^*$.])
// )

// #proof([])

= Integer Programming (IP)
== Definition

_Integer Programming_ is similar, but deals with *integer* variables, i.e. replacing the real variables in linear programming problem with integer variables:
$
#box(stroke: black, inset: 5pt, [$x in bb(R)^n$], baseline: 5pt) quad arrow.r quad #box(stroke: black, inset: 5pt, [$x in bb(Z)^n$], baseline: 5pt)
$
It is a non-convex optimization problem, and belongs to the complexity class of NP-complete.
For convex optimization problems, both the feasible region and the objective function are convex. However, for integer programming problems, the feasible region $bb(Z)^n$ is not convex.

== Example: Branching and cut for solving integer programming
Let us consider the following integer programming problem (source: #link("https://youtu.be/upcsrgqdeNQ?si=B5uilXqSrI5Jg976", "Youtube")):
$
  max_(x_1, x_2) quad &z = 5x_1 + 6x_2,\
  "s.t." quad &x_1 + x_2 <= 5,\
  &4x_1 + 7x_2 <= 28,\
  &x_1, x_2 >= 0,\
  &#box(stroke: black, inset: 5pt, [$x_1, x_2 in bb(Z)$])
$
where the last line is the integer constraint.
Or equivalently, in matrix form, we have
$
  c = vec(5, 6), quad
  A = mat(1, 1; 4, 7), quad
  b = vec(5, 28)
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
caption: [Solving an integer programming problem. The fesible region of linear programming problem is the green area, while the feasible region of integer programming problem are the discrete points marked by small circles.],
) <fig:integer-programming>

By relaxing the integer constraint, it becomes a linear programming problem that is easy to solve. The feasible region is the green polygon in @fig:integer-programming, and the optimal solution is the point $p_1$ at the line crossing.

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

#box(text(16pt)[```julia
# Sub-problem 1
using JuMP, HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4 * x1 + 7 * x2 <= 28)

@objective(model, Max, 5 * x1 + 6 * x2)

optimize!(model)
value(x1), value(x2), objective_value(model)
```
])

You can see the result of the linear programming problem.
#box(text(16pt)[```output
(2.3333333333333335, 2.6666666666666665, 27.666666666666668)
```])

==
#box(text(16pt)[```julia
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
```
])
By uncommenting the lines in the sub-problem, we can solve different sub-problems.
In practice, you do not need to write the branch and cut by yourself.
The solvers in `JuMP` package will do it for you.

#box(text(16pt)[```julia
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
])
You can see the result of the integer programming problem.
#box(text(16pt)[```output
(3.0, 2.0, 27.0)
```])

Note here, the `HiGHS` solver backend is used. Please refer to the #link("https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers")[documentation] to check alternative solvers supporting integer programming.

== Example 2: Spin glass problem
The spin glass energy function is defined as
$
E(sigma) = sum_(i < j) J_(i j) sigma_i sigma_j + sum_i h_i sigma_i
$
where $sigma_i = plus.minus 1$ is the spin of the $i$-th particle, $J_(i j)$ is the interaction strength between the $i$-th and $j$-th particles, and $h_i$ is the external magnetic field.

Finding a spin configuration with the minimum energy is a famous NP-complete problem.
At a first look, it is not an integer programming problem because there is a quadratic term in the energy function.
However, it turns out that the problem can be reduced to an integer programming problem by introducing a auxiliary variables.
Let us first introduce some boolean variables $s_i = (1 - sigma_i) \/ 2$.
Then we introduce auxiliary variables $d_(i j) = s_i plus.circle s_j$, which can be achieved by adding the following four linear constraints:
$
d_(i j) &<= s_i + s_j,\
d_(i j) &<= 2 - s_i - s_j,\
d_(i j) &>= s_i - s_j,\
d_(i j) &>= s_j - s_i.
$

Finally, we have the new representation of the energy function
$
E = sum_(i < j) J_(i j) (1 - 2 d_(i j)) + sum_i h_i (1 - 2 s_i),
$
which is linear in the variables $s_i$ and $d_(i j)$.

The Julia implementation is as follows:

#box(text(16pt)[```julia
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
])
The result is
#box(text(16pt)[```output
(-5.000000000000282, [1.0, 1.0000000000000648, 0.9999999999999196, -1.0, 0.9999999999999708, 1.0, -1.0, -1.0, -1.0, -1.0])
```
])

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

#definition([_(Tropical matrix factorization problem)_ Given an $m times n$ tropical matrix $C_(m times n)$ and an integer $k$, are there two tropical matrices $A_(m times k)$ and $B_(k times n)$ such that $C = A times.circle B$.])

This problem is NP-complete for $k >= 7$@Shitov2014. Even in the special case that all variables are either $-infinity$ or $0$, the problem is not easy. By mapping $-infinity$ to `false` and $0$ to `true`, we have the following boolean constraints:
$
C_(i j) = or.big_(l=1)^k A_(i l) and B_(l j).\
$

It can be solved by reducing to an integer programming problem.
To facilitate the reduction, we introduce a auxiliary tensor $D_(i l j) = A_(i l) and B_(l j)$. The resulting integer programming problem is as follows:

$
min quad &1\
"s.t." quad &D_(i l j) <= A_(i l)\
&D_(i l j) <= B_(l j) \
&D_(i l j) >= A_(i l) + B_(l j) - 1\
&C_(i j) <= sum_(l=1)^k D_(i l j)\ 
&D_(i l j) <= C_(i j)\ 
&A_(i l), B_(l j), D_(i l j) in {0,1}
$

Since we do not have any objective function, we set the 
The lines 2-4 encode the constraint $D_(i l j) = A_(i l) and B_(l j)$ with inequalities. The lines 5-6 encodes the tropical matrix multiplication constraints.

#box(text(16pt)[```julia
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
])

In this example, there are $m times k times n$ variables.
For such a large number of variables, the integer programming solver can handle a problem of size $50 times 6 times 50$ in a few seconds.

== Benchmarking


= Semidefinite Programming (SDP)
_Semidefinite programming_ is a generalization of linear programming. It is also a convex optimization problem, hence it is easy to solve.

Recall that a symmetric matrix $A in bb(R)^(n times n)$ is positive semidefinite (PSD) if $x^T A x >= 0$ for all $x in bb(R)^n$. This property is equivalent to:
1. $A$ has all non-negative eigenvalues.
2. $A$ can be written as $A = U^T U$ (Cholesky decomposition) for some $U in bb(R)^(n times n)$, i.e., $A_(i j) = u_i^T u_j$ where $u_i$ is the $i$-th column of $U$.

The goal of semidefinite programming is to solve optimization problems where the input is a matrix that is constrained to be PSD. I.e. we optimize over $X in bb(R)^(n times n)$ where $X in K$ and: $K = {M | M succ.eq 0}$.

#box(stroke: black, inset: 10pt, width: 100%, [Quiz: Show that $K$ is a convex set.])

_Semidefinite program (SDP)_. Let $f$ be a convex function. We seek to find $X in bb(R)^(n times n)$ which solves:
$
min quad &f(X)\
"s.t." quad &X succ.eq 0,\
&tr(A_i X) >= b_i, quad i = 1,...,k
$
Here $A_1,...,A_k$ and $b_1,...,b_k$ are input constraints. It is very common to have: $f(X) = tr(C X)$ for some $C$. I.e. to have our objective be a linear function in $X$. Let us vectorize $X$ as $x in bb(R)^(n^2)$ and compare the above problem with the LP, the only difference is that $X$ is constrained to be PSD instead of requiring every element of $X$ to be non-negative.
SDP is more general than LP because a linear programming problem can be viewed as a special case of semidefinite programming problem where $X$ is a diagonal matrix.

== Example 5: Spin-glass ground state problem
Let us consider obtaining an approximate solution to the spin-glass ground state problem.
The tightest known approximation ratio for the maximum cut problem (a special case of the spin-glass ground state problem) is 0.878,
which is achieved by using semidefinite programming @Goemans1995.
Recall that the spin-glass ground state problem is a quadratic integer programming problem that is difficult to solve exactly:
$
min quad &sum_(i j) J_(i j) sigma_i sigma_j\
"s.t." quad &sigma_i in {-1, 1}, forall i = 1, ..., n
$
where we do not consider the external magnetic field $h_i$ for simplicity.

To find an approximate solution of this problem, we first arrange the spin-spin correlation function into a matrix $X$ as follows:
$
X_(i j) = mat(sigma_1^2, sigma_1 sigma_2, ..., sigma_1 sigma_n; sigma_1 sigma_2, sigma_2^2, ..., sigma_2 sigma_n; dots, dots, dots, dots; sigma_1 sigma_n, sigma_2 sigma_n, ..., sigma_n^2)
$
It immediately follows that $X$ is a PSD matrix with the following constraints:
- The rank of $X$ is 1.
- $X$ is binary.
- The diagonal elements of $X$ are 1.
By relaxing the first two constraints, we have the SDP:
$
min quad & tr(J X)\
"s.t." quad &X succ.eq 0,\
&X_(i i) = 1, forall i = 1, ..., n
$ <eq:spin-glass-sdp>
This relaxation effectively generalizes the $sigma_i in {-1, 1}$ to a $n$-dimensional embedding on the unit sphere $x_i in bb(S)^n$ (@fig:spin-glass-sdp).
The spin correlation function $sigma_i sigma_j in {-1, 1}$ is mapped to the inner product of the embedding vectors $1 <= x_i^dagger x_j <= 1$, and $X_(i j) = x_i^dagger x_j.$
Given the optimal solution to $X$, we can recover $(x_1, ..., x_n)$ through the cholesky decomposition: $X = U^dagger U$ and set $x_i$ to be the $i$-th column of $U$ (blue arrow in @fig:spin-glass-sdp).
To obtain a binary solution $(sigma_1, ..., sigma_n)$, we map $(x_1, ..., x_n)$ to the binary values by introducing a *random* hyper-plane $H$ through the origin (the blue plane in @fig:spin-glass-sdp). If the $i$-th component of $u_1$ is "above" the hyper-plane, we set $sigma_i = 1$; otherwise, we set $sigma_i = -1$.

#figure(canvas({
  import draw: circle, line, rotate, content
  circle((0, 0), radius: 2.0)
  for (x, y, i) in ((-0.5, 1.2, 0), (1.2, 0.9, 1), (-1.2, -0.5, 2), (-0.5, -1.2, 3)){
    circle((x, y), radius: 0.1, fill: black, name: str(i))
  }
  for i in (0, 1, 2){
    content((rel: (0, 0.4), to: str(i)), [$sigma_#(i+1)$])
  }
  content((rel: (0.4, 0), to: "3"), [$sigma_4$])
  circle((0, 0), radius: 0.1, fill: blue, name: "o", stroke: none)
  line("0", "3", stroke: red)
  line("1", "2", stroke: red)
  line("1", "0", stroke: red)
  line("2", "3", stroke: red)
  line("2", "0", stroke: red)
  for i in range(4){
    line("o", str(i), mark: (end: "straight"), stroke: blue)
  }
  content((0.5, 1.5), "+1")
  content((-1.0, -1.0), "-1")
  rotate(40deg)
  circle((0, 0), radius: (0.2, 2.0), fill: aqua.transparentize(50%), stroke: (dash: "dashed"), name: "0")
}),
caption: [The hyper-plane $H$ (in blue) deciding the cut of graph $G = ({1, 2, 3, 4}, {(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)}).$ The edges are in red, the embedding of $sigma$ is in a hyper-sphere is denoted as blue vectors of unit length.]
) <fig:spin-glass-sdp>

The Julia code is as follows:
#box(text(16pt)[```julia
using JuMP, COSMO, Graphs, LinearAlgebra

function maxcut_sdp(G::SimpleGraph)
    n = nv(G)
    model = Model(COSMO.Optimizer)
    @variable(model, X[1:n, 1:n], PSD)
    for i in 1:n
        @constraint(model, X[i, i] == 1)
    end
    @objective(model, Min, sum(X[e.src, e.dst] for e in edges(G)))
    optimize!(model)
    return project_hyperplane(value(X), randn(n))
end

# `X` is the optimal solution of the SDP
# `H` is a vector orthogonal to the hyperplane
function project_hyperplane(X::AbstractMatrix{Float64}, H::Vector{Float64})
    n = length(H)
    @assert size(X, 1) == size(X, 2) == n
    # solve the Cholesky decomposition through eigen decomposition (stabler)
    res = eigen(X)
    U = Diagonal(sqrt.(max.(res.values, 0))) * res.vectors'
    return [dot(U[:, i], H) > 0 ? 1 : -1 for i in 1:n]
end

G = random_regular_graph(100, 3)
approx_maxcut = maxcut_sdp(G)

cut_size = sum(approx_maxcut[e.src] != approx_maxcut[e.dst] for e in edges(G))
```
])
A typical result is
#box(text(16pt)[```output
130
```])
It is very suppising that even the hyper-plane is randomly generated, the result is very close to the optimal solution: $135$, i.e. here $alpha approx 0.963$ is achieved.

_Analysis_. We are interested to know how good this approximate solution is in theory.
For simplicity, we consider the case that $J$ being an adjacency matrix of some graph $G = (V, E)$. Then finding the ground state energy is equivalent to finding the maximum cut of $G$.
#definition([_(Maximum cut problem)_ Given a graph $G = (V, E)$, the maximum cut problem is to find the largest set of edges $E' subset E$ such that $V$ can be partitioned into two subsets $V_1$ and $V_2$ such that $E' = {(i, j) | i in V_1, j in V_2}$.])

To connect the maximum cut problem with the anti-ferromagnetic spin-glass ground state problem, we use spin $sigma_i = +1$ and $-1$ to represent the $i$-th vertex being in $V_1$ and $V_2$ respectively. Each cut contributes $-2 J_(i j) sigma_i sigma_j$ to the energy function, where $sigma_i$ is the spin of the $i$-th vertex. Maximizing the cut size is equivalent to minimizing the energy function.

#theorem([There exists $alpha = 0.879$ with the following property. If $(x_1, ..., x_n)$ is optimal for the SDP for spin glass ground state problem, with an objective value $-overline("Maxcut")(G)$, $H$ is a random hyperplane through the origin, and $"Cut"(H)$ is the size of the edge cut consisting of those edges $(i, j) in E$ for which $x_i$ and $x_j$ are separated by $H$, then
$
bb(E)["Cut"(H)] >= alpha times overline("Maxcut")(G).
$
where $bb(E)$ is the expectation operator. Since $overline("Maxcut")(G)$ upper bounds the maximum cut size, $alpha$ is also an approximation ratio for the maximum cut problem.
])

#proof([
Let $theta_(i j) = arccos(x_i x_j), 0 <= theta_(i j) < pi$, #highlight([the probability that an edge is cut by $H$ is $theta_(i j)/pi$]). Then the expected cut size is
$
bb(E)["Cut"(H)] &= sum_(i,j in E) theta_(i j)/pi\
&>= sum_(i,j in E) 1/(beta pi) (1 - x_i x_j)\
&>= 2/(beta pi) times overline("Maxcut")(G) arrow.double.r alpha = 2/(beta pi)
$
])
From the first line to the second line, we have used an important observation that $1 - cos(theta)$ is upper bounded by $beta theta$ for some $beta > 0$ as shown in @fig:maxcut-sdp.
By solving the equation, we obtain $beta = 0.879$.

#figure(canvas({
import plot
  plot.plot(size: (8, 5),
    x-tick-step: 1,
    y-tick-step: 1,
    x-label: [$theta$],
    y-label: [],
    y-max: 3,
    y-min: 0,
    x-max: 2 * calc.pi,
    x-min: 0,
    name: "plot",
    {
      let f1 = x => 1 - calc.cos(x)
      let f2 = x => 0.733 * x
      plot.add(
        domain: (0, 2 * calc.pi),
        f1,
        label: [$1 - cos(x)$],
        style: (stroke: blue)
      )
      plot.add(
        domain: (0, 5),
        f2,
        label: [$ 0.7246 x$],
        style: (stroke: red)
      )
    }
  )
})
) <fig:maxcut-sdp>

= Hands-on

== (Integer programming) The minimum set cover problem
#definition([_(Minimum set cover problem)_ Given a set of elements $cal(S) = {1, 2, ..., n}$ and a collection of subsets $S_1, S_2, ..., S_m$ of $cal(S)$, the minimum set cover problem is to find the smallest collection of subsets that covers all elements in $cal(S)$.])

Consider the following example:
$
  cal(S) &= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},\
  S_1 &= {1, 2, 3, 4},\
  S_2 &= {2, 3, 4, 5},\
  S_3 &= {3, 4, 5, 6},\
  S_4 &= {1, 7, 8, 9},\
  S_5 &= {1, 4, 7, 10},\
  S_6 &= {2, 4, 6, 8, 10},\
  S_7 &= {1, 3, 5, 7, 9}
$
Use the integer programming to solve this problem.

_Solution_:
The minimum set cover problem can be formulated as an integer programming problem. The variables are $x = {x_1, x_2, ..., x_m}$, which are binary variables indicating whether the corresponding subset is included in the cover.
$
min_(x) & sum_(i=1)^m x_i\
"s.t." & (sum_(i : j in S_i) x_i) >= 1, forall j = 1, 2, ..., n\
& x_i in bb(Z)_2, forall i = 1, 2, ..., m
$

The Julia code is as follows:

#box(text(16pt)[```julia
using JuMP, HiGHS

function minimum_set_cover(n::Int, S::Vector{Vector{Int}})
    m = length(S)
    model = Model(HiGHS.Optimizer)
    @variable(model, x[1:m], Bin)

    for j in 1:n
        @constraint(model, sum(x[i] for i=1:m if j in S[i]) >= 1)
    end

    @objective(model, Min, sum(x))

    optimize!(model)
    return findall(value.(x) .> 0.5)
end

minimum_set_cover(10, [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [1, 7, 8, 9], [1, 4, 7, 10], [2, 4, 6, 8, 10], [1, 3, 5, 7, 9]])
```
])
The result is
#box(text(16pt)[```output
2-element Vector{Int64}:
 6
 7
```])
which means the minimum set cover is $S_6$ and $S_7$.

== SCIP: parameter tuning
https://www.scipopt.org/doc/html/PARAMETERS.php

== Hands-on: Integer programming for crystal structure prediction
Run the example in GitHub: https://github.com/Br0kenSmi1e/CrystalStructurePrediction.jl

#box(text(16pt)[```bash
git clone https://github.com/Br0kenSmi1e/CrystalStructurePrediction.jl
cd CrystalStructurePrediction.jl
cd ~/.julia/dev/CrystalStructurePrediction
make init  # initialize the project
make run-example  # run the SrTiO3 example
```
])
*It implements*: Gusev, V.V., et al., 2023. Optimality guarantees for crystal structure prediction. Nature 619, 68–72. https://doi.org/10.1038/s41586-023-06071-y

*Task*: Improve the performance of SCIP solver by tuning the parameters. Try pushing the lattice size to `(4, 4, 4)`.

==
#bibliography("refs.bib")

// = Appendix: Integer programming for error correction
// == Example 4.1: Code distance of linear codes
// In classical error correction theory, information are encoded into codewords. The _code distance_ is the minimum Hamming distance between any two codewords. For a linear code, the code distance equals to the minimum weight of the non-zero codewords. Deciding whether the code distance is at least $d$ is known to be in complexity class NP-complete@vardy1997intractability, which is unlikely to be solvable in time polynomial in the input size.

// A linear code can be defined by a parity check matrix $H in bb(F)^(m times n)_2$, where $bb(F)_2$ is the finite field with two elements $(0+0 = 1+1 = 0,1+0 =0+1= 1,1 times 0=0 times 1=0 times 0 = 0,1 times 1 =1)$. The codewords are 
// $
//   C = {x in bb(F)^n_2 | H x = 0},
// $
// and the code distance is
// $
//   d = min_(0 eq.not x in C) w(x)
// $
// where $w(x)$ is the weight of $x$, i.e. the number of non-zero elements in $x$.
// Take the Hamming code as an example, the parity check matrix is
// $
// H = mat(0,0,0,1,1,1,1; 0,1,1,0,0,1,1; 1,0,1,0,1,0,1).
// $
// It has $16$ codewords, which can encode $4$ bits into $7$ bits.
// The code distance of the Hamming code is $3$, since the minimum weight of the non-zero codewords is 
// $
//   w(mat(0,1,0,1,0,1,0)^T) = 3.
// $
// To find the code distance of a linear code, we can formulate it as:
// $
//   min quad & w(z)\
// "s.t." quad &H z = 0\
// &z eq.not 0\ 
// $
// It can be converted into an integer programming problem as follows:
// $
// min quad &sum^n_(i = 1) z_i\
// "s.t." quad & sum^n_(j = 1)H_(i j) z_j = 2 k_i quad triangle.small.r "equivalent to " H z = 0 "in" bb(F)^n_2\
// & sum^n_(i = 1) z_i >= 1\
// &z_j in {0,1}, k_i in bb(Z)
// $

// The Julia implementation is as follows:
// #box(text(16pt)[```julia
// using JuMP, HiGHS

// function code_distance(H::Matrix{Int}; verbose = false)
//     m,n = size(H)
//     model = Model(HiGHS.Optimizer)
//     !verbose && set_silent(model)

//     @variable(model, 0 <= z[i = 1:n] <= 1, Int)
//     @variable(model, 0 <= k[i = 1:m], Int)
    
//     for i in 1:m
//         @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i])
//     end
//     @constraint(model, sum(z[j] for j in 1:n) >= 1)

//     @objective(model, Min, sum(z[j] for j in 1:n))
//     optimize!(model)
//     @assert is_solved_and_feasible(model) "The problem is infeasible!"
//     return  objective_value(model)
// end

// H = [0 0 0 1 1 1 1;0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
// code_distance(H) == 3
// ```
// ])
// Here we verify that the code distance of the Hamming code is indeed $3$.

// == Example 4.2: Code distance of CSS quantum codes
// A CSS quantum code is quantum error correction code composed of $X$ stabilizers and $Z$ stabilizers that are characterized by two parity check matrices $H_x$ and $H_z$ respectively@calderbank1996good@steane1996multiple. The code distance of a CSS quantum code is the minimum of the code distance of the two classical codes defined by $H_x$ and $H_z$.
// $
//   d = min(d_x, d_z),
// $
// where $d_z$ is defined as
// $
//   d_z = min_( z in C_z \ exists overline(Z)_i , overline(Z)_i z eq.not 0) w(z),
// $

// where $C_z$ is the code space defined by $H_z$, $overline(Z)_i$ is the $i$-th logical $Z$ operator, and $w(z)$ is the weight of $z$, i.e. the number of non-zero elements in $z$.
// $d_x$ is defined similarly.
// Here, the constraint
// - $z in C_z$: $z$ satisfies all constraints specified by $Z$ stabilizers, i.e. no Pauli-$X$ error happens.
// - $overline(Z)_i z eq.not 0$: the logical $Z$ at $i$-th position does not commute with $z$, i.e. the logical state changes and the $overline(Z)_i$ operator value is changed. This constraint is the only difference compared to the classical code distance problem.
// // Compare to the classical code distance problem, the only difference is that we add a constraint $overline(Z)_i z eq.not 0$. Since the quantum logical $|overline(00000) angle.r$ is a superposition state, and we need to find a non-zero state $z$ not only in the code space but also in another logical space, like $|overline(01101) angle.r$. And logical $X$ and $Z$ operators are anti-commutative, we have at least one $overline(Z)_i z eq.not 0$.

// The above problem can be converted into an integer programming problem as follows@landahl2011fault@bravyi2024high:
// $
// min quad &sum^n_(i = 1) z_i\
// "s.t." quad & sum^n_(j = 1)H_(i j) z_j = 2 k_i quad triangle.small.r "equivalent to " H z = 0 "in" bb(F)^n_2\
// & sum^n_(j = 1) (overline(Z)_i)_j z_j = 2 l_j + r_j quad triangle.small.r "equivalent to " overline(Z)_i z = r_i "in" bb(F)^n_2\
// & sum^k_(i = 1) r_i >= 1\
// &z_j, r_j in {0,1}, k_i,l_j in bb(Z)
// $

// In the following, we use the Steane code as an example. The Steane code is constructed using two Hamming code for protecting against both $X$ and $Z$ errors. Here we verify that the code distance of the Steane code is indeed $3$ with JuMP:
// #box(text(16pt)[```julia
// using JuMP, HiGHS

// function code_distance(Hz::Matrix{Int},lz::Matrix{Int}; verbose = false)
//     m, n = size(Hz)
//     num_lz = size(lz, 1)
//     model = Model(HiGHS.Optimizer)
//     !verbose && set_silent(model)

//     @variable(model, 0 <= z[i = 1:n] <= 1, Int)
//     @variable(model, 0 <= k[i = 1:m], Int)
//     @variable(model, 0 <= l[i = 1:num_lz], Int)
//     @variable(model, 0 <= r[i = 1:num_lz] <= 1, Int)
    
//     for i in 1:m
//         @constraint(model, sum(z[j] for j in 1:n if Hz[i,j] == 1) == 2 * k[i])
//     end

//     for i in 1:num_lz
//         @constraint(model, sum(z[j] for j in 1:n if lz[i,j] == 1) == 2*l[i] + r[i])
//     end
//     @constraint(model, sum(r[i] for i in 1:num_lz) >= 1)

//     @objective(model, Min, sum(z[j] for j in 1:n))
//     optimize!(model)
//     @assert is_solved_and_feasible(model) "The problem is infeasible!"
//     return  objective_value(model)
// end

// using TensorQEC:logical_oprator,SteaneCode,CSSTannerGraph

// tanner = CSSTannerGraph(SteaneCode())
// lx,lz = logical_oprator(tanner)
// dz = code_distance(Int.(tanner.stgz.H), Int.(lz))
// dx = code_distance(Int.(tanner.stgx.H), Int.(lx))
// min(dz,dx) == 3
// ```
// ])

// == Example 4.3: Decoding for linear codes
// The decoding problem for linear codes is to find the most likely error pattern given the syndrome, where the syndrome is obtained from the stabilizer measurements. For a linear code defined by the parity check matrix $H$, the error pattern $e$ is related to the syndrome $s$ by $s = H e$.
// The decoding problem is to find the error pattern $e$ that is most likely to occur for any given syndrome $s$.
// Under the assumption that the most likely error pattern has the minimum weight, it can be formulated as an integer programming problem as follows@landahl2011fault:
// $
// min quad &sum^n_(i = 1) e_i\
// "s.t." quad & sum^n_(j = 1)H_(i j) e_j = 2 k_i + s_i quad triangle.small.r "equivalent to " H e = s "in" bb(F)^n_2\
// &e_j in {0,1}, k_i in bb(Z)
// $
// In the following, we use the Hamming code as an example to show how to decode the error pattern with JuMP:

// #box(text(16pt)[```julia
// using JuMP, HiGHS
// using TensorQEC: Mod2

// function ip_decode(H::Matrix{Int}, sydrome::Vector{Mod2}; verbose::Bool = false)
//     m,n = size(H)
//     model = Model(HiGHS.Optimizer)
//     !verbose && set_silent(model)

//     @variable(model, 0 <= z[i = 1:n] <= 1, Int)
//     @variable(model, 0 <= k[i = 1:m], Int)
    
//     for i in 1:m
//         @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (sydrome[i].x ? 1 : 0))
//     end

//     @objective(model, Min, sum(z[j] for j in 1:n))
//     optimize!(model)
//     @assert is_solved_and_feasible(model) "The problem is infeasible!"
//     return Mod2.(value.(z) .> 0.5)
// end

// H = Mod2[0 0 0 1 1 1 1; 0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
// error_bits = Mod2[1, 0, 0, 0, 0, 0, 0]  # Some random error pattern
// syd = H * error_bits

// ip_decode(Int.(H),syd) == error_bits
// ```
// ])

