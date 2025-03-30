#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations, coordinate
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#set math.mat(row-gap: 0.1em, column-gap: 0.7em)

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#let spinconfig(m, n, cfg, dx: 1, dy: 1, x:0, y:0) = {
  import draw: *
  let len = 0.5
  grid((x, y), (x + (m - 1) * dx, y + (n - 1) * dy), step: (dx, dy), stroke: (paint: gray, thickness: 0.5pt))
  for i in range(m) {
    for j in range(n) {
      if cfg.at(i * n + j) {
        line((x + i * dx, y + j * dy - len/2), (x + i * dx, y + j * dy + len/2), stroke: (paint: red, thickness: 2pt), mark: (end: "straight"))
      } else {
        line((x + i * dx, y + j * dy - len/2), (x + i * dx, y + j * dy + len/2), stroke: (paint: blue, thickness: 2pt), mark: (start: "straight"))
      }
    }
  }
}



#let triangle(Js, hs, colors: (blue, blue, red)) = {
  import draw: *
  let s(it) = text(16pt, it)
  for (i, (x, y, color, h)) in ((0, -1.5, colors.at(0), hs.at(0)), (0, 1.5, colors.at(1), hs.at(1)), (2.5, 0, colors.at(2), hs.at(2))).enumerate() {
    circle((x, y), radius: 0.6, fill: color.lighten(40%), name: "s" + str(i))
    content((x, y), s[#h])
  }
  line("s0", "s1", stroke: (paint: black, thickness: 1pt), name: "line1")
  line("s0", "s2", stroke: (paint: black, thickness: 1pt), name: "line2")
  line("s1", "s2", stroke: (paint: black, thickness: 1pt), name: "line3")
  content("line1.mid", box(fill: white, s[#Js.at(0)], inset: 0.1em))
  content("line2.mid", box(fill: white, s[#Js.at(1)], inset: 0.1em))
  content("line3.mid", box(fill: white, s[#Js.at(2)], inset: 0.1em))
}

#let pseudo_random(seed, n) = {
  // Simple linear congruential generator
  let a = 1664525
  let c = 1013904223
  let m = calc.pow(2, 32)
  let sequence = ()
  let x = seed
  for i in range(n) {
    x = calc.rem(a * x + c, m)
    sequence.push(x / m)  // normalize to [0,1]
  }
  sequence
}

#let random_numbers = pseudo_random(12345, 100)
#let random_bools(p) = random_numbers.map(x => x < p)


#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Simulated Annealing],
  subtitle: [],
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




= Spin glass

== The hardness
#align(center, box(stroke: black, inset: 0.5em, [
    Coupling $J_(i j)$ and bias $h_i$ freely tuned $arrow.r$ We get a spin glass!
  ])
)

Spin glass ground state finding problem is hard, it is NP-complete (hardest problems in NP), which is believed to be impossible to solve in polynomial time.

*NP problems*: Decision problems, features the property that given a solution, it is easy to verify whether the solution is correct in polynomial time.

#let alice(loc, rescale: 1, flip: false, label: none, words: none) = {
  import draw: *
  let r = 0.4 * rescale
  let xr = if flip { -r } else { r }
  circle(loc, radius: r, name: "alice")
  circle((rel: (xr * 0.3, 0.3 * r), to: loc), radius: (0.2 * r, 0.15 * r), name: "eye", stroke: none, fill: black)
  line((anchor: if flip { -70deg } else { -110deg }, name: "alice"), (rel: (-1.6 * xr, -3.5 * r), to: "alice"), (rel: (1.5 * xr, -3.7 * r), to: "alice"), (anchor: if flip { -120deg } else { -60deg }, name: "alice"), stroke: (paint: black, thickness: 1pt), name: "line1")
  line((anchor: 40%, name: "line1"), (loc.at(0) - xr, loc.at(1) - 5 * r))
  line((anchor: 57%, name: "line1"), (loc.at(0) + xr, loc.at(1) - 5 * r))
  hobby((anchor: if flip { 20deg } else { 160deg }, name: "alice"), (rel: (-2 * xr, 0), to: "alice"), (rel: (-xr * 1.5, -r), to: "alice"), rescale: 0.5)
  if label != none {
    content((loc.at(0), loc.at(1) - 2.5 * r), label)
  }
  if words != none {
    content((loc.at(0) + 5 * xr, loc.at(1) - 0.5 * r), box(width: rescale * 70pt, words))
  }
}

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
    content((loc.at(0) + 6 * xr, loc.at(1) - 1.5 * r), box(width: rescale * 70pt, words))
  }
}

#figure(canvas({
  import draw: *
  let s(it) = text(16pt, it)
  bob((0, 0), rescale: 2, flip: false, label: s[$n^100$], words: text(16pt)[Both of us are\ difficult to solve.])
  alice((15, 0), rescale: 2, flip: true, label: s[$1.001^n$], words: text(16pt)[Sorry, we are not in the same category.])
})) <fig:np-complete>


== NP problem hierarchy
#let pointer(start, end, angle: 45deg) = {
  import draw: *
  draw.get-ctx(ctx => {
    let (ctx, va) = coordinate.resolve(ctx, start)
    let (ctx, vb) = coordinate.resolve(ctx, end)
    let dy = vb.at(1) - va.at(1)
    let dx = dy / calc.tan(angle)
    let cx = va.at(0) + dx
    let cy = vb.at(1)
    line(start, (cx, cy))
    line((cx, cy), end)
  })
}
#grid(canvas(length: 0.7cm, {
  import draw: *
  let s(it) = text(14pt, it)
  circle((0, 0), radius: (6, 4), stroke: (paint: black, thickness: 1pt))
  circle((3, 0), radius: (3, 2), stroke: (paint: black, thickness: 1pt))
  hobby((-1, 1), (-2, 3), (-5, 3), (-7, 0), (-5, -3), (-2, -3), (-1, -1), close: true, smooth: 10pt, stroke: (paint: black, thickness: 1pt), fill: blue.transparentize(50%))
  circle((-4, 0), radius: (2, 2), stroke: (paint: black, thickness: 1pt), fill: yellow.transparentize(20%))

  content((1, 3), s[NP])
  content((-4, 0), s[P])
  content((-2, 2), s[BQP])
  content((3, 0), s[NP-complete])
  for (i, j, name) in ((-2, -1.5, "B"), (5.5, 0, "C"), (-6.5, 0, "S"), (-1.5, 3, "G")) {
    circle((i, j), radius:0.2, fill: black, name:name)
  }
  content((-7, -4), box(s[Factoring], inset: 5pt), name: "Factoring")
  content((-8.5, 0), box(s[Quantum \ Sampling], inset: 5pt), name: "Sampling")
  content((0, 5), box(s[Spin \ Glass], inset: 5pt), name: "Spinglass")
  content((-7, 5), box(s[Graph \ isomorphism], inset: 5pt), name: "GI")
  set-style(stroke: (paint: black, thickness: 1pt))
  pointer("B", "Factoring", angle: 45deg)
  pointer("C", "Spinglass", angle: -65deg)
  pointer("G", "GI", angle: -65deg)
}),
[#box([*P:* Polynomial time solvable\
*NP:* Polynomial time verifiable\
*NP-complete:* The hardest problems in NP\
\
*BQP:* Polynomial time solvable on a quantum computer
], width: 350pt)],
columns: 2, gutter: 20pt)

- NP: the problem set that can be solved by a "magic coin" - a coin that gives the best outcome with probability 1, i.e. it is non-deterministic.

== NP-complete problems can be reduced to each other

- Reduction: Problem $A$ can be reduced to problem $B$ if $A$ can be "solved by" solving $B$.

#align(center, canvas(length: 1.2cm, {
  import draw: *
  for (x, y, txt, color) in (
      (5, -1, "Independent Set", white),
      (-4, -1, "QUBO (Spin Glass)", white),
      (5, 1, "Set Packing", white),
      (-5, 3, "Dominating Set", white),
      (-8, -1, "Max Cut", white),
      (-8, 3, "Coloring", white),
      (-5, 1, "k-SAT", white),
      (0, -1, "Circuit SAT", white),
      (5, 3, "Vertex Matching", gray),
      (3, -3, "Independent Set on KSG", white),
      (-4, -3, "QUBO on Grid", white),
      (0, 1, "Integer Factorization", gray),
      (1, 3, "Vertex Cover", white),
      (-2, 3, "Set Cover", white)
    ){
    content((x, y), box(text(14pt, txt), stroke:black, inset:7pt, fill:color.lighten(50%)), name: txt)
  }
  let arr = "straight"
  for (a, b, markstart, markend, color) in (
    ("Integer Factorization", "Circuit SAT", none, arr, black),
    ("Set Packing", "Independent Set", arr, arr, black),
    ("k-SAT", "Independent Set", none, arr, black),
    ("Independent Set on KSG", "Independent Set", arr, arr, black),
    ("Integer Factorization", "Circuit SAT", none, arr, black),
    ("Vertex Cover", "Set Cover", none, arr, black),
    ("Dominating Set", "k-SAT", arr, none, black),
    ("Coloring", "k-SAT", arr, none, black),
    ("Circuit SAT", "k-SAT", arr, none, black),
    ("Set Packing", "Independent Set", arr, arr, black),
    ("k-SAT", "Independent Set", none, arr, black),
    ("Circuit SAT", "QUBO (Spin Glass)", none, arr, black),
    ("Integer Factorization", "Independent Set on KSG", none, arr, black),
    ("k-SAT", "Circuit SAT", arr, none, black),
    ("Independent Set on KSG", "QUBO on Grid", arr, none, black),
    ("QUBO (Spin Glass)", "Max Cut", arr, arr, black),
    ("Vertex Matching", "Set Packing", none, arr, black),
    ("Vertex Cover", "Independent Set", arr, none, black),
    ("QUBO on Grid", "QUBO (Spin Glass)", arr, none, black),
  ){
    line(a, b, mark: (end: markend, start: markstart), stroke: color)
  }
  rect((-6, -4), (6, -2), stroke:(dash: "dashed"))
  content((0, -4), box(fill: white, inset: 7pt)[Low-dimensional topology])
}))



== A generic reduction strategy

#figure(canvas(length: 1.3cm,{
  import draw: *
  let s(it) = text(14pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  content((0, 0), boxed(s[Problem]), name: "problem")
  content((0, -2), boxed(s[Spin glass]), name: "spin-glass")
  content((5, -2), boxed(s[Ground state]), name: "ground-state")
  content((5, 0), boxed(s[Solution]), name: "solution")
  line("problem", "spin-glass", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line1")
  line("spin-glass", "ground-state", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line2")
  line("ground-state", "solution", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line3")
  content((rel: (-1.0, 0), to: "line1.mid"), s[Reduction])
  content((rel: (0.0, -0.3), to: "line2.mid"), s[Cooling])
  content((rel: (1.0, 0), to: "line3.mid"), s[Extraction])
}))

1. Given a problem $A$ in NP, we construct its verification circuit $C_A$ (take $A$ and a solution $bold(s)$ as input, output 1 if $bold(s)$ is a solution to $A$, otherwise output 0) of polynomial size.
2. Represent the verification circuit $C_A$ as a spin glass $S(C_A)$ with polynomial size.
3. Fix the output spin to logic state $1$.
4. Find the ground state of $S(C_A)$ using a generic spin glass solver.
5. Read out the spin configuration associated with $bold(s)$ from the ground state.


== Logic circuit to spin glass
#figure(table(columns: 4, table.header([Gate], [Gadget], [Ground states], [Lowest energy]),
[Logical not: $not$], [
  #canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(11pt, it)
  for (i, (x, y, color)) in ((-1.5, 0, blue), (1.5, 0, red)).enumerate() {
    circle((x, y), radius: 0.6, fill: color.lighten(40%), name: "s" + str(i))
  }
  line("s0", "s1", stroke: (paint: black, thickness: 1pt), name: "line1")
  content("line1.mid", box(fill: white, s[1], inset: 0.1em))
})

],
[(-1, +1), (+1, -1)], [-1],
[Logical and: $and$], [#canvas(length: 0.6cm, {
  triangle((1, -2, -2), (1, 1, -2))
})
],
[(-1, -1, -1), (+1, -1, +1),\ (-1, +1, +1), (+1, +1, +1)], [-3],
[Logical or: $or$], [
#canvas(length: 0.6cm, {
  triangle((1, -2, -2), (-1, -1, 2))
})

],
[(-1, -1, -1), (+1, -1, -1),\ (-1, +1, -1), (+1, +1, +1)], [-3],
))
The spin glass gadget for logic gates@Gao2024. The blue/red spin is the input/output spins. The numbers on the vertices are the biases $h_i$ of the spins, the numbers on the edges are the couplings $J_(i j)$.

== Example: Encoding the factoring problem to a spin glass
We introduce how to convert the factoring problem into a spin glass problem.
Factoring problem is the cornerstone of modern cryptography, it is the problem of given a number $N$, find two prime numbers $p$ and $q$ such that $N = p times q$.

== Composibility of logic gadgets

An implmentation of NAND operation through composing the logic $and$ and $not$ gadgets.
#figure(canvas({
  import draw: *
  let s(it) = text(16pt, it)
  triangle((1, -2, -2), (1, 1, -2))

  for (i, (x, y, color, t)) in ((2.5, 0, white, "2"), (5, 0, red, none)).enumerate() {
    circle((x, y), radius: 0.6, fill: color.lighten(40%), name: "s" + str(i))
    content((x, y), box(s[#t], inset: 0.1em))
  }
  line("s0", "s1", stroke: (paint: black, thickness: 1pt), name: "line1")
  content("line1.mid", box(fill: white, s[1], inset: 0.1em))
  content((-1, 1.5), s[$x$])
  content((-1, -1.5), s[$y$])
  content((7, 0), s[$not (x and y)$])
  content((2.5, -2.5), s[Biases added up], name: "add")
  line("add", (2.5, -1), mark: (end: "straight"))
}))

If and only if all gadgets have correct assignments, the energy of the combined gadget is minimized.

== Julia implementation of the reduction
#box(text(16pt)[```julia
source_problem = Factoring(3, 2, 15)

# Construct the spin glass
paths = reduction_paths(Factoring, SpinGlass)
mapres = reduceto(paths[1], source_problem)
target_problem(mapres) |> num_variables  # output: 63

# Extract the ground state
ground_state = read_config(solve(target_problem(mapres), SingleConfigMin())[])

# Verify the solution
solution = extract_solution(mapres, ground_state) # output: [1, 0, 1, 1, 1]
ProblemReductions.read_solution(source_problem, solution) # output: (5, 3)
```
])

== Simulate the cooling process to find the ground state?
- Fact 1: Spin-glass can encode any problem in NP, including the famous factoring problem: $N = p times q$.
- Fact 2: A physical system thermalizes through the Hamiltonian dynamics
  $
    cases((dif bold(q))/(dif t) = (partial H)/(partial bold(p)), (dif bold(p))/(dif t) = - (partial H)/(partial bold(q)))
  $
  where $H = T + V$ is the Hamiltonian, $bold(q)$ and $bold(p)$ are the generalized position and momentum variables.
- Fact 3: The thermal state at zero temperature is the ground state.
- Fact 4: If we drive the spin glass to the ground state, we can read out the solution from the spin configuration.

== Simulated Annealing

Simulated annealing is an algorithm for finding the *global optimum* of a given function, which mimics the physical process of cooling down a material.
#algorithm(
  {
    import algorithmic: *
    Function("SimulatedAnnealing", args: ([$bold(s)$], [$T_"init"$], [$alpha$], [$n_"steps"$]),
    {
    For(cond: [$i = 1 dots n_"steps"$], {
        Assign([$T$], [$alpha^i T_"init"$])
        // Choose a random spin to flip
        Assign([$bold(s)'$], [$bold(s)$ with random spin flipped])
          // Accept with probability $e^{-\Delta E/T}$ if energy increases
        State([$r ~ cal(U)(0,1) quad$  #Ic[random number]]) 
        If(cond: [$r < e^(-(H(bold(s)') - H(bold(s)))\/T)$], {
          Assign([$bold(s)$], [$bold(s)'$])
        })
      // Decrease temperature according to cooling schedule
    })
    
    Return([$bold(s)$])
    }
    )
  }
)
$T_"init"$ is the initial temperature, $alpha < 1$ is the cooling rate, $n_"steps"$ is the number of steps.

== Simulated Annealing

The temperature determining the probability distribution of states through the Boltzmann statistics:
$
  p(bold(s)) ~ e^(-H(bold(s))\/T)
$
At the thermal equilibrium, the system effectively finds the distribution with the lowest free energy:
$F = angle.l H angle.r - T S$, where $S$ is the entropy.
- When the temperature $T$ is high, the system tends to find the distribution with large entropy, making the system behave more randomly.
- As the temperature decreases, the system tends to find the distribution with lower energy, making the system more likely to be in the low-energy states.

== Cooling schedule
*Key assumption*: The system stays in the thermal equilibrium.

- Cool too quickly, the system may get trapped in a local minimum.
- Cool too slowly, the algorithm becomes computationally expensive.

Common cooling schedules include:

1. Linear: $T_k = T_"init" - k dot (T_"init" - T_"final")/n$
2. Exponential: $T_k = T_"init" dot alpha^k$ where $0 < alpha < 1$
3. Logarithmic: $T_k = T_"init"/log(k+1)$

- _Remark_: Theoretically, with a logarithmic cooling schedule that decreases slowly enough, simulated annealing will converge to the global optimum with probability 1.

== The landscape matters
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((0, 0), (3, 3))
  circle((1.5, 2.2), radius: (0.5, 0.5), fill: blue.lighten(60%), stroke: none)
  circle((1.5, 0.8), radius: (0.5, 0.5), fill: blue.lighten(60%), stroke: none)
  
  let dx = 8
  rect((dx, 0), (dx + 3, 3))
  let n = 20
  for i in range(n) {
    circle((dx + 0.5 + 2 * random_numbers.at(i), 0.5 + 2 * random_numbers.at(i+n)), radius: (random_numbers.at(i+2*n))/4, fill: blue.lighten(60%), stroke: none)
  }
  content((dx + 1.5, -0.8), s[Frustrated, Glassy\ Many local minima])
  content((1.5, -0.8), s[No frustration, Ordered,\ Countable local minima])

  content((dx/2 + 1.5, 1.5), s[Add more "triangles"])
  content((dx/2 + 1.5, 1.1), s[$arrow.double.r$])
}))

== Spin dynamics

Adiabatic simulated bifurcation (aSB) dynamics@Goto2021:
$
  V_("aSB") = sum_(i=1)^N (x_i^4 / 4 + (a_0 - a(t))/2 x_i^2)- c_0 sum_(i < j) J_(i j) x_i x_j\
  H_("aSB") = a_0/2 sum_(i=1)^N p_i^2 + V_("aSB")\
  dot(p)_i = - (partial V_("aSB"))/(partial x_i)\
  dot(x)_i = (partial H_("aSB"))/(partial p_i)
$

== References
#bibliography("refs.bib")