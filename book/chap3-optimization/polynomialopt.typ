#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#show: book-page.with(title: "Polynomial optimization")

= Integer programming

Let us consider the following integer programming problem (source: #link("https://youtu.be/upcsrgqdeNQ?si=B5uilXqSrI5Jg976", "Youtube")):
$
  z = 5x_1 + 6x_2\
  x_1 + x_2 <= 5\
  4x_1 + 7x_2 <= 28\
  x_1, x_2 >= 0\
  x_1, x_2 in bb(Z)
$

If we relax the integer constraint, we get the following linear programming problem:
$
  z = 5x_1 + 6x_2\
  x_1 + x_2 <= 5\
  4x_1 + 7x_2 <= 28\
  x_1, x_2 >= 0
$
which is a convex optimization problem.

Q: how to show this is a convex optimization problem?

#figure(canvas(length:0.9cm, {
  import plot
  import draw: *
  plot.plot(size: (10,7),
    x-tick-step: none,
    y-tick-step: none,
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
      plot.add(
        domain: (0, 5),
        f1,
        label: [$x_1 + x_2 = 5$],
      )

      plot.add(
        domain: (0, 5),
        f2,
        label: [$4x_1 + 7x_2 = 28$],
      )
      plot.add-fill-between(
        domain: (0, 5),
        x => calc.min(f1(x), f2(x)),
        x => 0,
        label: [feasible region],
      )
      plot.add(
        domain: (0, 5),
        f3,
        label: [objective function],
        style: (stroke: (paint: black, dash: "dashed")),
      )
 
      plot.add(range(36).map(x => (calc.rem(x, 6), calc.div-euclid(x, 6))), style: (stroke: none), mark: "o")
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
  content((rel: (0, 0.4), to: "plot.p1"), [1])
  circle("plot.p2", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p2"), [2])
  circle("plot.p3", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p3"), [3])
  circle("plot.p5", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p5"), [5])
  circle("plot.p6", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p6"), [6])
  circle("plot.p7", radius: 0.1, fill: red)
  content((rel: (0, 0.4), to: "plot.p7"), [7])
}),
) <fig:integer-programming>

For convex optimization problems, both the feasible region and the objective function are convex. However, for integer programming problems, the feasible region is not convex.

== Linear programming
```julia
# Sub-problem 1
using JuMP, HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4x1 + 7x2 <= 28)

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

@objective(model, Max, 5x1 + 6x2)

optimize!(model)
value(x1), value(x2), objective_value(model)
```
You can see the result of the optimization problem.
```output
(2.3333333333333335, 2.6666666666666665, 27.666666666666668)
```

Then we collect the integer constraints back by adding some inequalities.

#align(center, canvas({
  import draw: *
  let boxed(c) = box(stroke: black, inset: (x: 10pt, y: 10pt), c)
  content((0, 0), boxed([Sub-problem 1\
  - $z = 27.67$\
  - $x_1 = 2.33$\
  - $x_2 = 2.67$]), name: "sub-problem-1")
  content((-4, -4), boxed([Sub-problem 2\
  - $z = 27$\
  - $x_1 = 3$\
  - $x_2 = 2$]), name: "sub-problem-2")
  content((4, -4), boxed([Sub-problem 3\
  - $z = 26.75$\
  - $x_1 = 1.75$\
  - $x_2 = 3$]), name: "sub-problem-3")
  content((2, -8), boxed([Sub-problem 4\
  - infeasible]), name: "sub-problem-4")

  content((6, -8), boxed([Sub-problem 5\
  - $z = 25.57$\
  - $x_1 = 1$\
  - $x_2 = 3.42$]), name: "sub-problem-5")

  content((4, -12), boxed([Sub-problem 6\
  - $z = 24$\
  - $x_1 = 0$\
  - $x_2 = 4$]), name: "sub-problem-6")

  content((8, -12), boxed([Sub-problem 7\
  - $z = 23$\
  - $x_1 = 1$\
  - $x_2 = 3$]), name: "sub-problem-7")

  line("sub-problem-1", "sub-problem-2", mark: (end: "straight"), name: "l12")
  content((rel: (-0.8, 0), to: "l12.mid"), [$x_2 <= 2$])
  line("sub-problem-1", "sub-problem-3", mark: (end: "straight"), name: "l13")
  content((rel: (0.8, 0), to: "l13.mid"), [$x_2 >= 3$])
  line("sub-problem-3", "sub-problem-4", mark: (end: "straight"), name: "l34")
  content((rel: (-0.8, 0), to: "l34.mid"), [$x_1 >= 2$])
  line("sub-problem-3", "sub-problem-5", mark: (end: "straight"), name: "l35")
  content((rel: (0.8, 0), to: "l35.mid"), [$x_1 <= 1$])
  line("sub-problem-5", "sub-problem-6", mark: (end: "straight"), name: "l56")
  content((rel: (-0.8, 0), to: "l56.mid"), [$x_2 >= 4$])
  line("sub-problem-5", "sub-problem-7", mark: (end: "straight"), name: "l57")
  content((rel: (0.8, 0), to: "l57.mid"), [$x_2 <= 3$])
}))

== Spin glass problem

== Hard instance
```julia
using JuMP, HiGHS, TropicalNumbers

function tropical_svd(C::Matrix{TropicalAndOr}, k::Int; verbose = false)
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
```

= Polynomial optimization

== Moment

- A measure can be characterized by expectation values.