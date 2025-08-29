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
    title: [Mathematical Optimization],
    subtitle: [Linear Programming and Integer Programming],
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


#outline-slide()

= Introduction to optimization
== Convex optimization
*Optimization* is the study of minimizing an *objective function* over a *feasible region*.
*Convex*: $f(a x + (1-a) y) <= a f(x) + (1-a) f(y)$, e.g. a quadratic function $f(x) = x^2$ is convex, but $f(x) = x^3$ is not.

#figure(
  canvas(length: 0.8cm, {
    import draw: *
    let f1 = x => x * x
    let f2 = x => calc.pow(x, 4) / 4 - calc.pow(x, 2) / 2
    plot.plot(
      name: "plot",
      size: (6, 4),
      x-label: none,
      y-label: none,
      x-tick-step: 2,
      y-tick-step: 2,
      legend: "north",
      {
        plot.add(
          f1,
          domain: (-3, 3),
          samples: 100,
          label: text(14pt)[Convex: $f(x) = x^2$]
        )
        let xmid = -2 + 3 * 0.3
        plot.add-anchor("ca", (-2, 4))
        plot.add-anchor("cb", (1, 1))
        plot.add-anchor("yt", (xmid, 1 + 3 * 0.7))
        plot.add-anchor("ft", (xmid, f1(xmid)))
    })
      
    // Draw points on the convex curve
    circle("plot.ca", radius: 0.1, fill: blue)
    circle("plot.cb", radius: 0.1, fill: blue)
    
    // Draw the line segment connecting the points
    line("plot.ca", "plot.cb", stroke: (dash: "dashed"))
    
    circle("plot.yt", radius: 0.1, fill: black)
    circle("plot.ft", radius: 0.1, fill: blue)
    
    // Draw a vertical line to show the difference
    line("plot.yt", "plot.ft", stroke: (dash: "dotted"))

    set-origin((8.5, 0))
    plot.plot(
      name: "plot",
      size: (6, 4),
      x-label: none,
      y-label: none,
      x-tick-step: 2,
      y-tick-step: 10,
      legend: "north",
      {
        plot.add(
          f2,
          domain: (-1.5, 1.5),
          samples: 100,
          label: text(14pt)[Non-convex: $f(x) = x^4\/4 - x^2\/2$]
        )
        let a = -0.5
        let b = 1
        let xmid = a * 0.7 + b * 0.3
        plot.add-anchor("ca", (a, f2(a)))
        plot.add-anchor("cb", (b, f2(b)))
        plot.add-anchor("yt", (xmid, 0.7 * f2(a) + f2(b) * 0.3))
        plot.add-anchor("ft", (xmid, f2(xmid)))
    })
    // Draw points on the convex curve
    circle("plot.ca", radius: 0.1, fill: blue)
    circle("plot.cb", radius: 0.1, fill: blue)
    
    // Draw the line segment connecting the points
    line("plot.ca", "plot.cb", stroke: (dash: "dashed"))
    
    circle("plot.yt", radius: 0.1, fill: black)
    circle("plot.ft", radius: 0.1, fill: blue)
    
    // Draw a vertical line to show the difference
    line("plot.yt", "plot.ft", stroke: (dash: "dotted"))


  })
)
Q: What is the feasible region of the above two functions?

== Convex objective function is not enough
Convex optimization:
- The objective function is convex.
- The feasible region is convex.

Q: Is the following feasible region convex?
#figure(canvas({
  import draw: *
  rect((-2, -2), (2, 2))
  circle((0, 0), radius: 1, fill: blue, stroke: none)
  content((0, -2.5), text(14pt)[A])

  set-origin((5, 0))
  rect((-2, -2), (2, 2))
  circle((0, -1), radius: 0.5, fill: blue, stroke: none)
  circle((0, 1), radius: 0.5, fill: blue, stroke: none)
  content((0, -2.5), text(14pt)[B])

  set-origin((5, 0))
  rect((-2, -2), (2, 2))
  line((-1, -1), (-1, 1), (1, 1), (1, -1), (0, 0), close: true, fill: blue, stroke: none)
  content((0, -2.5), text(14pt)[C])

  set-origin((5, 0))
  rect((-2, -2), (2, 2))
  line((-1, -1), (-1, 1), (1, -1), close: true, fill: blue, stroke: none)
  content((0, -2.5), text(14pt)[D])
}))

== Convex optimization problems are easy to solve

#align(center, table(inset: 9pt,
  columns: (auto, auto, auto, auto),
  table.header([*Julia Package*], [*Features*], [*Algorithms*], [*Applications*]),
  table.cell(fill:yellow)[#link("https://github.com/jump-dev/JuMP.jl", "JuMP.jl")\ @Dunning2017], table.cell(fill:yellow)[Mathematical optimization modeling], table.cell(fill:yellow)[Linear programming\ Integer programming\ SDP, etc.], table.cell(fill:yellow)[Combinatorial optimization, planning and scheduling],
  [#link("https://github.com/JuliaNLSolvers/Optim.jl", "Optim.jl")\ @Mogensen2018], [Gradient and non-gradient based optimization], [L-BFGS, Newton's method, Nelder-Mead method, etc.], [General purpose optimization],
  [#link("https://github.com/FluxML/Optimisers.jl", "Optimisers.jl")], [Robust to noise, cheap to compute], [Stochastic gradient descent (Adam, AdaGrad, RMSProp, etc.)], [Training neural networks],
))

= Linear Programming (LP)
_Linear Programming_ deals with the problem of optimizing a
- *linear objective function*
- subject to *linear equality and inequality constraints*
- on *real* variables.
$
  max_(x) quad &z = c^T x,\
  "s.t." quad &A x <= b,\
  &x >= 0\
  &#box(stroke: black, inset: 5pt, [$x in bb(R)^n$])
$ <eq:linear-programming>
where $c in bb(R)^n$ is the objective function coefficient vector, $A in bb(R)^(m times n)$ is the constraint matrix, and $b in bb(R)^m$ is the constraint vector.

== Remarks
- Linear programming is convex, hence it is easy to solve.
- The positivity constraint $x >= 0$ is not absolutely necessary. Adding this extra constraint does not sacrifice the generality of linear programming, because any linear program can be written into a canonical form by shifting the variables.

== Example 1: The Diet Problem
In the diet model, a list of available foods is given together with the nutrient content and the cost per unit weight of each food. A certain amount of each nutrient is required per day. For example, here is the data corresponding to two types of food (fish and rice) and three types of nutrients (starch, proteins, vitamins) in unit of kg:
#align(center, table(
  columns: (auto, auto, auto, auto, auto),
  table.header([], [Starch], [Proteins], [Vitamins], [Cost (RMB/kg)]),
  [fish], [0], [4], [2], [6],
  [rice], [7], [2], [1], [3.5],
))

- *Requirement*: per day of starch, proteins and vitamins is 8, 15 and 3 respectively.
- *Objective*: find how much of each food to consume per day so as to get the required amount per day of each nutrient at minimal cost.

== Solution
In the diet problem, a very natural choice of decision variables is:
- $x_1$: number of units of fish to be consumed per day,
- $x_2$: number of units of rice to be consumed per day.

This diet problem can therefore be formulated by the following linear program:
$
min_(x_1, x_2) & z = 6x_1 + 3.5x_2\
"s.t." & 0x_1 + 7x_2 ≥ 8\
& 4x_1 + 2x_2 ≥ 15\
& 2x_1 + x_2 ≥ 3\
& x_1 ≥ 0, x_2 ≥ 0.
$

==


In Julia programming language, we can solve the integer programming problem by using #link("https://github.com/jump-dev/JuMP.jl", "JuMP") package.
#figure(image("images/jump.png", width: 2cm, alt: "JuMP"))
#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: the diet problem])
}))

== Algorithms for solving linear programming problems

#let s(item) = table.cell(align: left)[#item]
#align(center, text(16pt, table(inset: 9pt,
  columns: (auto, auto, auto),
  table.header([*Algorithm*], [*Advantages*], [*Disadvantages*]),
  [*Simplex Method* \ @Dantzig1963], s[
    - Very efficient in practice
    - Finds exact solutions
    - Works well for sparse problems
    - Easy to warm-start from previous solutions
  ], s[
    - Exponential worst-case complexity
    - Can be inefficient for dense problems
    - Vulnerable to cycling in degenerate cases
  ],
  [*Interior Point Methods* \ @Karmarkar1984], s[
    - Polynomial worst-case complexity
    - Excellent for large, dense problems
    - More predictable solution times
  ], s[
    - Less efficient for sparse problems
    - Harder to warm-start
    - More complex implementation
  ],
  [*Ellipsoid Method* \ @Khachiyan1979], s[
    - Polynomial worst-case complexity
    - Theoretically important
    - Can handle some non-differentiable objectives
  ], s[
    - Very slow in practice
    - Rarely used for actual computation
  ],
)))

// === Simplex Method

// The Simplex Method, developed by George Dantzig in 1947, works by:
// 1. Starting at a vertex of the feasible region
// 2. Moving along edges to adjacent vertices that improve the objective value
// 3. Terminating when no further improvement is possible

// While it has exponential worst-case complexity, the Simplex Method is remarkably efficient for most practical problems and remains widely used today.

// === Interior Point Methods

// Interior Point Methods follow a different approach:
// 1. Start from a point inside the feasible region
// 2. Follow a path through the interior toward the optimal solution
// 3. Approach the boundary of the feasible region as the algorithm progresses

// These methods gained popularity in the 1980s and offer polynomial-time complexity, making them particularly valuable for large-scale problems.

== Formulating spin glass as a "linear" programming problem
The spin glass energy function is defined as
$
E(sigma) = sum_(i < j) J_(i j) sigma_i sigma_j + sum_i h_i sigma_i
$
where $sigma_i = plus.minus 1$ is the spin of the $i$-th particle, $J_(i j)$ is the interaction strength between the $i$-th and $j$-th particles, and $h_i$ is the external magnetic field.

== Handle the quadratic term
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

== Hey, it is not possible!

- Spin glass is in the complexity class of NP-complete.
- Linear programming is convex, hence it is easy to solve.

Q: Why there is a "contradiction"?

= Integer Programming (IP)
== Definition

_Integer Programming_ is similar to linear programming, but deals with *integer* variables, i.e. replacing the real variables in linear programming problem with integer variables:
$
#box(stroke: black, inset: 5pt, [$x in bb(R)^n$], baseline: 5pt) quad arrow.r quad #box(stroke: black, inset: 5pt, [$x in bb(Z)^n$], baseline: 5pt)
$
- _Remark_: Integer programming is a non-convex optimization problem, and belongs to the complexity class of NP-complete. The feasible region $bb(Z)^n$ is not convex.

== Example 2: Branching and cut for solving integer programming
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
) <fig:integer-programming>


Solving an integer programming problem. The feasible region of linear programming problem is the green area, while the feasible region of integer programming problem are the discrete points marked by small circles.

== Branch and cut
Step 1. Relaxing the integer constraint, it becomes a linear programming problem that is easy to solve.

// 2. $p_1$ is not feasible for the integer programming problem due to $x_2$ being non-integer. To force the variables to be integer, we add some inequalities constraints to the linear programming problem as shown in @fig:branching-and-cutting. Since $x_2$ is non-integer, we create two sub-problems (or branches) by adding two inequalities constraints $x_2 <= 2$ and $x_2 >= 3$ to the linear programming problem.

// 3. It turns out the sub-problem 2 with $x_2 <= 2$ accepts integer solution $p_2$ as the optimal solution. So we stop this branch and continue to solve the sub-problem 3 with $x_2 >= 3$.

== Branch and cut
#figure(canvas(length: 0.8cm, {
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
) <fig:branching-and-cutting>
2. "Branch" and "cut" over the non-integer variables by adding more constraints.

== Julia implementation

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: branch and cut for solving integer programming])
}))


== Comparison of Integer Programming Solvers

#align(center, text(16pt, table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  align: center,
  table.header(
    [*Solver*], 
    [*License Type*], 
    [*Performance*], 
    [*Features*], 
    [*Best For*]
  ),
  [HiGHS], 
  [Open Source (MIT)], 
  [Good], 
  [LP, MIP, QP support, Dual simplex, Interior point], 
  [Academic use, Open source projects],
  
  [SCIP], 
  [Academic free, Commercial paid], 
  [Very good], 
  [LP, MIP, MINLP, Branch-and-cut, Cutting planes], 
  [Research, Complex constraint programming],
  
  [Gurobi], 
  [Commercial (free academic)], 
  [Excellent], 
  [LP, MIP, MIQP, MIQCP, Parallel solving, Advanced heuristics], 
  [Industry applications, Large-scale problems],
  
  [CPLEX], 
  [Commercial (free academic)], 
  [Excellent], 
  [LP, MIP, QP, MIQP, Network flows, Distributed optimization], 
  [Enterprise applications, Integration with IBM tools]
)))

== Choice of solver
The choice of solver depends on your specific needs:
- For open-source projects or educational purposes, HiGHS and SCIP are good starting points.
- For industrial applications with large-scale problems, commercial solvers like Gurobi or CPLEX typically provide better performance and support.

All these solvers can be used with JuMP in Julia through their respective packages.

Please refer to the #link("https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers")[documentation] to check alternative solvers supporting integer programming.

== Example 3: Solving spin glass problem with integer programming

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: solving spin glass problem with integer programming])
}))


== Benchmarking


= Semidefinite Programming (SDP)
_Semidefinite programming_ is a generalization of linear programming. It is also a convex optimization problem, hence it is easy to solve.

Recall that a symmetric matrix $A in bb(R)^(n times n)$ is positive semidefinite (PSD) if $x^T A x >= 0$ for all $x in bb(R)^n$. This property is equivalent to:
1. $A$ has all non-negative eigenvalues.
2. $A$ can be written as $A = U^T U$ (Cholesky decomposition) for some $U in bb(R)^(n times n)$, i.e., $A_(i j) = u_i^T u_j$ where $u_i$ is the $i$-th column of $U$.

The goal of semidefinite programming is to solve optimization problems where the input is a matrix that is constrained to be PSD. I.e. we optimize over $X in bb(R)^(n times n)$ where $X succ.eq 0$.

#box(stroke: black, inset: 10pt, width: 100%, [Q: Show that $X$ is a convex set.])

_Semidefinite program (SDP)_. Let $f$ be a convex function. We seek to find $X in bb(R)^(n times n)$ which solves:
$
min quad &f(X)\
"s.t." quad &X succ.eq 0,\
&tr(A_i X) >= b_i, quad i = 1,...,k
$
Here $A_1,...,A_k$ and $b_1,...,b_k$ are input constraints. It is very common to have: $f(X) = tr(C X)$ for some $C$. I.e. to have our objective be a linear function in $X$.
- _Remark_: SDP is more general (and is harder) than LP because a linear programming problem can be viewed as a special case of semidefinite programming problem where $X$ is a diagonal matrix.

== Example 4: Approximating the spin-glass ground state

*Max-Cut Problem (a special case of the spin-glass ground state problem)*: Given a graph $G = (V, E)$, find a partition of the vertices $V$ into two sets $S$ and $V \\ S$ such that the number of edges crossing the partition is maximized.

#let show-graph-maxcut(vertices, edges_1, edges_2, cutted, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for k in cutted {
    let (i, j) = vertices.at(k)
    circle((i, j), radius:radius, name: str(k), fill:red)
  }
  for (k, l) in edges_1 {
    line(str(k), str(l))
  }
  for (k, l) in edges_2 {
    line(str(k), str(l), stroke: (paint: blue, thickness: 1pt, dash: "dashed"))
  }
}
#let vrotate(v, theta) = {
  let (x, y) = v
  return (x * calc.cos(theta) - y * calc.sin(theta), x * calc.sin(theta) + y * calc.cos(theta))
}

#figure(canvas(length: 1.0cm, {
  import draw: *
  // petersen graph
  let vertices1 = range(5).map(i=>vrotate((0, 2), i*72/180*calc.pi))
  let vertices2 = range(5).map(i=>vrotate((0, 1), i*72/180*calc.pi))
  let edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (6, 8), (7, 9), (8, 5), (9, 6))

  let edges_1 = ((2, 3), (0, 5), (9, 6))
  let edges_2 = ((0, 1), (1, 2), (3, 4), (4, 0), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (6, 8), (7, 9), (8, 5))

  show-graph-maxcut((vertices1 + vertices2).map(v=>(v.at(0) + 4, v.at(1)+4)), edges_1, edges_2, (1, 4, 7, 8), radius:0.2)
  hobby((2, 2.8), (4, 2.8), (5, 2.8), (6, 5.3), (5, 3.8), (4, 3.8), (3, 3.8), (2.5, 5), (2.5, 6), stroke: blue)
}))

Imagine we're organizing a party and need to divide people into two rooms. Some pairs of people like each other (connected by an edge), and some don't. To maximize drama and entertainment, we want to place as many friends as possible in different rooms - maximizing the number of friendships that cross between rooms.

== Max-Cut formulated as a spin-glass ground state problem
Mathematically, we can formulate this as:
$
max sum_(i,j in E) (1 - sigma_i sigma_j)/2
$

where $sigma_i in {-1, 1}$ indicates which room person $i$ is assigned to. When friends $i$ and $j$ are in different rooms ($sigma_i != sigma_j$), we get a contribution of 1 to our objective.


This is equivalent to minimizing:
$
min sum_(i,j in E) sigma_i sigma_j
$

which is exactly the spin glass ground state problem with $J_(i j) = 1$ for $(i,j) in E$ and $J_(i j) = 0$ otherwise.

== Relaxation to SDP

*Goal*: obtaining an approximate solution to the max-cut problem through _relaxation_ to semidefinite programming (SDP).

- _Remark_: SDP gives the tightest known approximation ratio for the maximum cut problem, which is 0.878 @Goemans1995. This algorithm is known as the Goemans-Williamson algorithm.

== SDP relaxation

The relaxation of the max-cut problem is:
$ sigma_i in {-1, 1} arrow.double.r x_i in bb(S)^n $
where $bb(S)$ is the unit sphere in $bb(R)^n$.

To find an approximate solution of this problem, we first arrange the spin-spin correlation function into a matrix $X$ as follows:
$
X_(i j) = mat(sigma_1^2, sigma_1 sigma_2, ..., sigma_1 sigma_n; sigma_1 sigma_2, sigma_2^2, ..., sigma_2 sigma_n; dots, dots, dots, dots; sigma_1 sigma_n, sigma_2 sigma_n, ..., sigma_n^2)
$
where the product $sigma_i sigma_j$ is replaced by the inner product $x_i^dagger x_j$.

==

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

== Recovering the binary solution
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
) <fig:spin-glass-sdp>

The hyper-plane $H$ (in blue) deciding the cut of graph $G = ({1, 2, 3, 4}, {(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)}).$ The edges are in red, the embedding of $sigma$ is in a hyper-sphere is denoted as blue vectors of unit length.

==

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: solving spin glass problem with SDP])
}))

==

#theorem([There exists $alpha = 0.879$ with the following property. If $(x_1, ..., x_n)$ is optimal for the SDP for spin glass ground state problem, with an objective value $-overline("Maxcut")(G)$, $H$ is a random hyperplane through the origin, and $"Cut"(H)$ is the size of the edge cut consisting of those edges $(i, j) in E$ for which $x_i$ and $x_j$ are separated by $H$, then
$
bb(E)["Cut"(H)] >= alpha times overline("Maxcut")(G).
$
where $bb(E)$ is the expectation operator. Since $overline("Maxcut")(G)$ upper bounds the maximum cut size, $alpha$ is also an approximation ratio for the maximum cut problem.
])

==
#proof([
Let $theta_(i j) = arccos(x_i x_j), 0 <= theta_(i j) < pi$, #highlight([the probability that an edge is cut by $H$ is $theta_(i j)/pi$]). Then the expected cut size is
$
bb(E)["Cut"(H)] &= sum_(i,j in E) theta_(i j)/pi\
&>= sum_(i,j in E) 1/(beta pi) (1 - x_i x_j)\
&>= 2/(beta pi) times overline("Maxcut")(G) arrow.double.r alpha = 2/(beta pi)
$
])
==
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

== Choose the right solver

When solving semidefinite programming problems like the SDP relaxation for MaxCut, choosing the right solver is crucial.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: center,
    [*Solver*], [*License*], [*Features*], [*Precision*],
    [Mosek], [Commercial (free for academics)], [Highly optimized, interior point method], [High],
    [COSMO\ @Garstka2021], [Apache 2.0 (open source)], [Conic programming, first order method], [Medium],
    [Clarabel\ @Goulart2024], [Apache 2.0 (open source)], [Interior point method], [High],
  ),
)

= Hands-on

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

== SCIP: performance and parameter tuning

#figure(image("images/ipbenchmark.png", width: 80%, alt: "IP benchmark"))

The SCIP @Achterberg2009 is open source, licensed under the Apache License 2.0. It provides more freedom for parameter tuning: https://www.scipopt.org/doc/html/PARAMETERS.php

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

