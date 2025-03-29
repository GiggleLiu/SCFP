#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, decorations, plot
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#show: book-page.with(title: "Simulated annealing")
#show: thmrules
#set math.equation(numbering: "(1)")

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)
#let proposition = thmbox("proposition", "Proposition", inset: (x: 1.2em, top: 1em), base: none)
#let theorem = thmbox("theorem", "Theorem", base: none)
#let proof = thmproof("proof", "Proof")
#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#let jinguo(c) = text(red)[[JG: #c]]

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

#let norm(v) = calc.sqrt(v.map(x => calc.pow(x, 2)).sum())
#let distance(a, b) = norm(a.zip(b).map(x => x.at(0) - x.at(1)))
#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#let udg-graph(vertices, unit:1) = {
  let edges = ()
  for (k, va) in vertices.enumerate() {
    for (l, vb) in vertices.enumerate() {
      if l < k and distance(va, vb) <= unit {
        edges.push((k, l))
      }
    }
  }
  return edges
}

#let draw_cube(cube_center, size, perspective: 0.3) = {
  import draw: *
    // Define the vertices of the front face
  let front_vertices = (
    (cube_center.at(0) - size/2, cube_center.at(1) - size/2),  // bottom-left
    (cube_center.at(0) + size/2, cube_center.at(1) - size/2),  // bottom-right
    (cube_center.at(0) + size/2, cube_center.at(1) + size/2),  // top-right
    (cube_center.at(0) - size/2, cube_center.at(1) + size/2)   // top-left
  )
  
  // Define the vertices of the back face
  let back_vertices = front_vertices.map(v => (
    v.at(0) + perspective, 
    v.at(1) - perspective
  ))
  
  // Draw the front face with named vertices
  for i in range(4) {
    circle(front_vertices.at(i), radius: 0.0, name: "f" + str(i))
    circle(back_vertices.at(i), radius: 0.0, name: "b" + str(i))
  }
  
  // Draw the back face with named vertices
  for i in range(4) {
    line("f" + str(i), "f" + str(calc.rem(i + 1, 4)))
    line("b" + str(i), "b" + str(calc.rem(i + 1, 4)))
  }
  
  // Connect front to back
  for i in range(4) {
    line("f" + str(i), "b" + str(i))
  }
}

#align(center, [= Simulated Annealing\
_Jin-Guo Liu_])


= Simulated annealing

== Spin glass and NP-complete problems

When the coupling can be freely tuned, the ferromagnetic Ising model becomes a spin glass.
Spin glass ground state finding problem is hard, it is NP-complete, i.e. if you can cool down a spin glass system to the ground state in polynomial time, you can solve any problem in NP in polynomial (to problem size) time, which is believed to be impossible.
NP problems are decision problems, features the property that given a solution, it is easy to verify whether the solution is correct in polynomial time.

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
    content((loc.at(0) + 5 * xr, loc.at(1) - 0.5 * r), box(width: rescale * 7em, words))
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
    content((loc.at(0) + 6 * xr, loc.at(1) - 1.5 * r), box(width: rescale * 7em, words))
  }
}

#figure(canvas(length: 1.2cm, {
  import draw: *
  let s(it) = text(0.8em, it)
  bob((0, 0), rescale: 1, flip: false, label: s[$n^100$], words: text(1em)[Both of us are difficult to solve.])
  alice((8, 0), rescale: 1, flip: true, label: s[$1.001^n$], words: text(1em)[Sorry, we are not in the same category.])
}),
caption: [There is a huge barrier between problems can and cannot be solved in time polynomial in the problem size.]
) <fig:np-complete>

== Spin glass is NP-complete
We start by showing the following statement is true: _If you can drive a Spin glass system to the ground state, you can prove any theorem._ Note that theorem proving is not different from other problems in NP. First it is a decision problem, second, with a proof, it is easy to verify whether the solution is correct in polynomial time. A spin glass system can encode a target problem by tuning its couplings $J_(i j)$ and biases $h_i$ in the Hamiltonian.
After driving the spin glass system to the ground state, we can read out the proof from the spin configuration.

#figure(canvas(length: 1.3cm, {
  import draw: *
  let s(it) = text(1em, it)
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

In the following, we show how to construct a spin glass system that can encode any logic circuit. The ground state of a spin glass can encode the truth table of any logic circuit. In @tbl:logic-circuit-to-spin-glass, we show the spin glass gadget for well-known logic gates $not$, $and$ and $or$. These logic gates can be used to construct any logic circuit.

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
[(-1, +1),\ (+1, -1)], [-1],
[Logical and: $and$], [#canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(11pt, it)
  for (i, (x, y, color, h)) in ((0, -1.5, blue, 1), (0, 1.5, blue, 1), (2.5, 0, red, -2)).enumerate() {
    circle((x, y), radius: 0.6, fill: color.lighten(40%), name: "s" + str(i))
    content((x, y), s[#h])
  }
  line("s0", "s1", stroke: (paint: black, thickness: 1pt), name: "line1")
  line("s0", "s2", stroke: (paint: black, thickness: 1pt), name: "line2")
  line("s1", "s2", stroke: (paint: black, thickness: 1pt), name: "line3")
  content("line1.mid", box(fill: white, s[1], inset: 0.1em))
  content("line2.mid", box(fill: white, s[-2], inset: 0.1em))
  content("line3.mid", box(fill: white, s[-2], inset: 0.1em))
})
],
[(-1, -1, -1),\ (+1, -1, +1),\ (-1, +1, +1),\ (+1, +1, +1)], [-3],
[Logical or: $or$], [
#canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(11pt, it)
  for (i, (x, y, color, h)) in ((0, -1.5, blue, -1), (0, 1.5, blue, -1), (2.5, 0, red, 2)).enumerate() {
    circle((x, y), radius: 0.6, fill: color.lighten(40%), name: "s" + str(i))
    content((x, y), s[#h])
  }
  line("s0", "s1", stroke: (paint: black, thickness: 1pt), name: "line1")
  line("s0", "s2", stroke: (paint: black, thickness: 1pt), name: "line2")
  line("s1", "s2", stroke: (paint: black, thickness: 1pt), name: "line3")
  content("line1.mid", box(fill: white, s[1], inset: 0.1em))
  content("line2.mid", box(fill: white, s[-2], inset: 0.1em))
  content("line3.mid", box(fill: white, s[-2], inset: 0.1em))
})

],
[(-1, -1, -1),\ (+1, -1, -1),\ (-1, +1, -1),\ (+1, +1, +1)], [-3],
), caption: [The spin glass gadget for logic gates@Gao2024. The blue/red spin is the input/output spins. The numbers on the vertices are the biases $h_i$ of the spins, the numbers on the edges are the couplings $J_(i j)$.]) <tbl:logic-circuit-to-spin-glass>

With these gadgets, we can construct any logic circuits utilizing the composibility of logic gadgets. Given two logic gadgets $H_1$ and $H_2$, in the ground state of the combined gadget $H_1 compose H_2$, both $H_1$ and $H_2$ are in their own ground state. i.e. the logic expressions associated with $H_1$ and $H_2$ are both satisfied. Therefore, the ground state of $H_1 compose H_2$ is the same as the truth table of the composed logic circuit.

== Example: Encoding the factoring problem to a spin glass
We introduce how to convert the factoring problem into a spin glass problem.
Factoring problem is the cornerstone of modern cryptography, it is the problem of given a number $N$, find two prime numbers $p$ and $q$ such that $N = p times q$.


```julia
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

In this example, we map a integer factoring problem (solving which efficiently indicates a breakdown of RSA cryptography) to a spin glass, the resulting spin glass has $63$ spins.
Then we resort to the generic tensor network solver to find the ground state.
The solution can be read out from the spin configuration with the help of intermediate information provided by the mapping procedure.
Again, the tensor network based method can not handle large systems.
In the following, we will introduce a physics-inspired algorithm, the simulated annealing, to find the ground state of a spin glass.

== Simulated Annealing
Simulated annealing is an algorithm for finding the global optimum of a given function, which mimics the physical process of cooling down a material.
#block(stroke: black, inset: 0.5em, [
  *Can we simulate the cooling process to find the ground state of a spin glass?*
  - Fact 1: Spin-glass can encode any problem in NP, including the famous factoring problem: $N = p times q$.
  - Fact 2: A physical system thermalizes through the Hamiltonian dynamics
    $
      cases((dif bold(q))/(dif t) = (partial H)/(partial bold(p)), (dif bold(p))/(dif t) = - (partial H)/(partial bold(q)))
    $
    where $H = T + V$ is the Hamiltonian, $bold(q)$ and $bold(p)$ are the generalized position and momentum variables.
  - Fact 3: The thermal state at zero temperature is the ground state.
  - Fact 4: If we drive the spin glass to the ground state, we can read out the solution from the spin configuration.
])


A cooling process is characterized by lowering the temperature $T$ from a high initial temperature $T_"init"$ to a low final temperature $T_"final"$ following a cooling schedule. The temperature determining the probability distribution of states through the Boltzmann statistics:
$
  p(bold(s)) ~ e^(-H(bold(s))\/T)
$
At the thermal equilibrium, the system effectively finds the distribution with the lowest free energy:
$F = angle.l H angle.r - T S$, where $S$ is the entropy. When the temperature $T$ is high, the system tends to find the distribution with large entropy, making the system behave more randomly. As the temperature decreases, the system tends to find the distribution with lower energy, making the system more likely to be in the low-energy states. This transition can not happen abruptly, otherwise the system will get stuck in a local minimum. We have to wait the dynamics to thermalize the system.

The algorithm works as follows:

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
The algorithm starts with an initial spin configuration $bold(s)$, initial temperature $T_"init"$ and cooling rate $alpha < 1$. It then iteratively updates the spin configuration with probability $e^(-Delta E\/T)$. This update strategy can be carefully designed to achieve shorter mixing time. Here, we use the simplest single spin flip strategy.

The cooling schedule is also crucial for the algorithm's performance. If we cool too quickly, the system may get trapped in a local minimum. If we cool too slowly, the algorithm becomes computationally expensive. Common cooling schedules include:

- Linear: $T_k = T_"init" - k dot (T_"init" - T_"final")/n$
- Exponential: $T_k = T_"init" dot alpha^k$ where $0 < alpha < 1$
- Logarithmic: $T_k = T_"init"/log(k+1)$

Theoretically, with a logarithmic cooling schedule that decreases slowly enough, simulated annealing will converge to the global optimum with probability 1. However, in practice, faster cooling schedules are often used with good empirical results.

== Parallel tempering

The energy landscape of a spin glass is rough, with many local minima separated by high barriers. Normal Metropolis sampling is inefficient at low temperatures, as the system gets trapped in a local minimum.

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
  content((dx + 1.5, -0.8), s[Glassy, Many local minima])
  content((1.5, -0.8), s[Ordered, Countable local minima])

  content((dx/2 + 1.5, 1.5), s[Add more frustration])
  content((dx/2 + 1.5, 1.1), s[$arrow.double.r$])
}))

Parallel tempering (also known as replica exchange) is a Monte Carlo method designed to improve sampling efficiency for systems with rough energy landscapes, such as spin glasses. The key idea is to simulate multiple replicas of the system at different temperatures simultaneously, allowing configurations to be exchanged between temperatures. It improves the exploration of the state space, and makes the sampling of low-energy states thermalize faster. In parallel tempering:

1. We simulate $M$ replicas of the system at different temperatures $T_1 < T_2 < ... < T_M$
2. Each replica evolves according to standard Metropolis dynamics at its temperature
3. Periodically, we attempt to swap configurations between adjacent temperature levels. The swap between configurations at temperatures $T_i$ and $T_(i+1)$ is accepted with probability:
  $
  P_"swap"(bold(s)_i, bold(s)_(i+1)) = min(1, e^(-(beta_i - beta_(i+1))(H(bold(s)_(i+1)) - H(bold(s)_i))))
  $
  where $beta_i = 1/T_i$ and $bold(s)_i$ is the configuration at temperature $T_i$.

== References
#bibliography("refs.bib")
