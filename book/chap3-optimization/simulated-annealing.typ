#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, decorations, plot, coordinate
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


#align(center, [= Simulated Annealing and Spin Dynamics\
_Jin-Guo Liu_])


= The spin glass ground state problem

The spin glass Hamiltonian is given by
$
  H = sum_(i < j) J_(i j) sigma_i sigma_j + sum_i h_i sigma_i
$ <eq:spin-glass-hamiltonian>
where $sigma_i$ is the spin of the $i$-th spin.
Spin glass ground state finding problem is hard, it is NP-complete (hardest problems in NP), which is believed to be impossible to solve in time polynomial in the problem size. In computational complexity, problems solvable in polynomial time are fundamentally different from those can not.

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
    content((loc.at(0), loc.at(1) - 2.8 * r), label)
  }
  if words != none {
    content((loc.at(0) + 5 * xr, loc.at(1) - 0.5 * r), box(width: rescale * 100pt, words))
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
    content((loc.at(0) + 6 * xr, loc.at(1) - 1.5 * r), box(width: rescale * 100pt, words))
  }
}

#figure(canvas(length: 1.2cm, {
  import draw: *
  let s(it) = text(11pt, it)
  bob((0, 0), rescale: 1, flip: false, label: s[$n^100$], words: s[Both of us are difficult to solve.])
  alice((8, 0), rescale: 1, flip: true, label: s[$1.001^n$], words: s[Sorry, we are not in the same category.])
}),
caption: [There is a huge barrier between problems can and cannot be solved in time polynomial in the problem size.]
) <fig:np-complete>

Among those polynomial time solution have not been found, the NP category is the most interesting one. They are the set of problems that can be solved by a _non-deterministic Turing machine_ - a hypothetical machine that always go to the correct branch and find the solution in polynomial time. If the machine spits out a solution, we can verify whether the solution is correct in polynomial time. For example, when we ask whether the Hamiltonian in @eq:spin-glass-hamiltonian has a ground state with energy lower than $E_0$, we can verify whether a spin configuration satisfies the condition immediately.
#align(center, box(stroke: black, inset: 0.5em, [
  Easy to verify $!=$ easy to solve (we believe)
]))
Easy to verify does not mean easy to solve, we do not know any algorithm that can find a such a spin configuration in polynomial time in the worst case.

Another example is the integer factorization problem, which is the foundation of RSA encryption. Its verification is easy, given two integers $a$ and $b$, we can verify whether $a times b$ equals $c$ in polynomial time.
```julia
c = BigInt(21267647932558653302378126310941659999)
@test a * b == c  # find a and b
```
However, its solving is hard, we do not know any algorithm that can find the two integers that multiply to $c$ in polynomial time in the worst case.

Even worse, although we know some problems are hard, we do not have a valid strategy to prove any problem in NP-complete can not be solved in time polynomial in the problem size. Let us denote those solvable problems in polynomial time as $P$. Whether $"P" = "NP"$ is still an open question. To characterize the hardness of problems, researchers create a complexity hierarchy based on the reduction relation.
If a problem $A$ can be solved by solving problem $B$, and the overhead is polynomial in the problem size, we say $A$ is reducible to $B$ and write $ A <=_p B, quad quad "A can be solved by solving B". $
Although we do not know a problem is absolutely hard, we know some problems are not easier than others.
With this in mind, we can create a hierarchy of computational hardness in @fig:np-complete-hierarchy.

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
#figure(canvas(length: 0.7cm, {
  import draw: *
  let s(it) = text(12pt, it)
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
caption: [Complexity classes of computational problems. *NP* is the set of decision problems that can be solved by a "magic coin" - a coin that gives the best outcome with probability 1, i.e. it is non-deterministic. *P* is the set of problems that can be solved in polynomial time. *NP-complete* is the set of hardest problems in NP. *BQP* is the set of problems that can be solved in polynomial time on a quantum computer.]
) <fig:np-complete-hierarchy>

NP-complete problems are the hardest problems in NP, all problems in NP, including themselves, can be reduced to any NP-complete problem in polynomial time, hence solving any NP-complete problem in polynomial time would solve all NP problems in polynomial time.
In @fig:np-complete-reduction, we show the reduction from some NP-complete problems to each other. Most of the reduction rules have already been implemented in the Julia package #link("https://github.com/GiggleLiu/ProblemReduction.jl", [ProblemReduction.jl]).


#figure(canvas(length: 1cm, {
  import draw: *
  for (x, y, txt, color) in (
      (5, -1, "Independent Set", white),
      (-4, -1, "QUBO (Spin Glass)", white),
      (5, 1, "Set Packing", white),
      (-5, 3, "Dominating Set", white),
      (-8, -1, "Max Cut", white),
      (-8, 3, "Coloring", white),
      (-5, 1, "k-SAT", white),
      (0, -1, "Circuit SAT", yellow),
      (5, 3, "Vertex Matching", gray),
      (3, -3, "Independent Set on KSG", white),
      (-4, -3, "QUBO on Grid", white),
      (0, 1, "Integer Factorization", gray),
      (1, 3, "Vertex Cover", white),
      (-2, 3, "Set Cover", white)
    ){
    content((x, y), box(text(12pt, txt), stroke:black, inset:5pt, fill:color.lighten(50%)), name: txt)
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
}), caption: [NP-complete problems can be reduced to each other in polynomial time.]) <fig:np-complete-reduction>

== Spin glass is NP-complete
To show a problem is NP-complete, we need to notice the fact that the *circuit SAT problem is the hardest problem in NP*. Circuit SAT is the problem of determining whether a given logic circuit has a satisfying assignment. That logic circuit includes a circuit that verifies a solution to a problem in NP, i.e. a circuit that takes a solution as input and outputs 1 if the solution is correct, otherwise 0.
Then we fix the output of the circuit to 1, and the problem becomes finding a satisfying assignment for the circuit. Since the size of the circuit is polynomial in the size of the original problem, if we can find a satisfying assignment for the circuit in polynomial time, we can find a solution to the original problem in polynomial time.

#figure(canvas(length: 1.2cm, {
  import draw: *
  let s(it) = text(11pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  content((0, 0), boxed(s[Problem in NP]), name: "problem")
  content((0, -2), boxed(s[Verification circuit]), name: "verification-circuit")
  content((5, -2), boxed(s[Input]), name: "input")
  content((5, 0), boxed(s[Solution]), name: "solution")
  line("problem", "verification-circuit", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line1")
  line("verification-circuit", "input", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line2")
  line("input", "solution", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "line3")
  content((rel: (-1.0, 0), to: "line1.mid"), s[Reduction])
  content((rel: (0.0, -0.3), to: "line2.mid"), s[Circuit SAT])
  content((rel: (1.0, 0), to: "line3.mid"), s[Extraction])
}),
caption: [A reduction from a problem in NP to the circuit SAT problem. The circuit SAT problem is the hardest problem in NP. If we can find a satisfying assignment for the circuit in polynomial time, we can find a solution to the original problem in polynomial time.]
)

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

In the following, we show an implementation of NAND operation through composing the logic $and$ and $not$ gadgets.
#figure(canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(11pt, it)
  triangle((1, -2, -2), (1, 1, -2))

  for (i, (x, y, color, t)) in ((2.5, 0, white, "-2"), (5, 0, red, none)).enumerate() {
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

== Example: Encoding the factoring problem to a spin glass
We introduce how to convert the factoring problem into a spin glass problem.
Factoring problem is the cornerstone of modern cryptography, it is the problem of given a number $N$, find two prime numbers $p$ and $q$ such that $N = p times q$.

#let multiplier-block(loc, size, sij, cij, pij, qij, pijp, qipj, sipjm, cimj) = {
  import draw: *
  rect((loc.at(0) - size/2, loc.at(1) - size/2), (loc.at(0) + size/2, loc.at(1) + size/2), stroke: black, fill: white)
  circle((loc.at(0) + size/2, loc.at(1) - size/2), name: sij, radius: 0)
  circle((loc.at(0) - size/2, loc.at(1) - size/4), name: cij, radius: 0)
  circle((loc.at(0) - size/2, loc.at(1) + size/4), name: qipj, radius: 0)
  circle((loc.at(0), loc.at(1) + size/2), name: pij, radius: 0)
  circle((loc.at(0) + size/2, loc.at(1) + size/4), name: qij, radius: 0)
  circle((loc.at(0), loc.at(1) - size/2), name: pijp, radius: 0)
  circle((loc.at(0) - size/2, loc.at(1) + size/2), name: sipjm, radius: 0)
  circle((loc.at(0) + size/2, loc.at(1) - size/4), name: cimj, radius: 0)
}

#let multiplier(m, n, size: 1) = {
  import draw: *
  for i in range(m){
    for j in range(n) {
      multiplier-block((-2 * i, -2 * j), size, "s" + str(i) + str(j), "c" + str(i) + str(j), "p" + str(i) + str(j), "q" + str(i) + str(j), "p" + str(i) + str(j+1) + "'", "q" + str(i+1) + str(j) + "'", "s" + str(i+1) + str(j - 1) + "'", "c" + str(i - 1) + str(j) + "'")
    }
  }
  for i in range(m){
    for j in range(n){
      if (i > 0) and (j < n - 1) {
        line("s" + str(i) + str(j), "s" + str(i) + str(j) + "'", mark: (end: "straight"))
      }
      if (i < m - 1){
        line("c" + str(i) + str(j), "c" + str(i) + str(j) + "'", mark: (end: "straight"))
      }
      if (j > 0){
        line("p" + str(i) + str(j), "p" + str(i) + str(j) + "'", mark: (start: "straight"))
      }
      if (i > 0){
        line("q" + str(i) + str(j), "q" + str(i) + str(j) + "'", mark: (start: "straight"))
      }
    }
  }
  for i in range(m){
    let a = "p" + str(i) + "0"
    let b = (rel: (0, 0.5), to: a)
    line(a, b, mark: (start: "straight"))
    content((rel: (0, 0.3), to: b), text(12pt)[$p_#i$])


    let a2 = "s" + str(i+1) + str(-1) + "'"
    let b2 = (rel: (-0.4, 0.4), to: a2)
    line(a2, b2, mark: (start: "straight"))
    content((rel: (-0.2, 0.2), to: b2), text(12pt)[$0$])

    let a3 = "s" + str(i) + str(n - 1)
    let b3 = (rel: (0.4, -0.4), to: a3)
    line(a3, b3, mark: (end: "straight"))
    content((rel: (0.2, -0.2), to: b3), text(12pt)[$m_#(i+m - 1)$])

  }
  for j in range(n){
    let a = "q0" + str(j)
    let b = (rel: (0.5, 0), to: a)
    line(a, b, mark: (start: "straight"))
    content((rel: (0.3, 0), to: b), text(12pt)[$q_#j$])

    let a2 = "q" + str(m) + str(j) + "'"
    let b2 = (rel: (-0.5, 0), to: a2)
    line(a2, b2, mark: (end: "straight"))


    let a3 = "c" + str(-1) + str(j) + "'"
    let b3 = (rel: (0.5, 0), to: a3)
    line(a3, b3, mark: (start: "straight"))
    content((rel: (0.3, 0), to: b3), text(12pt)[$0$])
  
    if (j < n - 1) {
      let a4 = "c" + str(m - 1) + str(j)
      let b4 = "s" + str(m) + str(j) + "'"
      bezier(a4, b4, (rel: (-1, 0), to: a4), (rel: (-0.5, -1), to: a4), mark: (end: "straight"))
    } else {
      let a4 = "c" + str(m - 1) + str(j)
      line(a4, (rel: (-0.5, 0), to: a4), mark: (end: "straight"))
      content((rel: (-0.8, 0), to: a4), text(12pt)[$m_#(j+m)$])
    }
    if (j < n - 1) {
      let a5 = "s0" + str(j)
      let b5 = (rel: (0.4, -0.4), to: a5)
      line(a5, b5, mark: (end: "straight"))
      content((rel: (0.2, -0.2), to: b5), text(12pt)[$m_#j$])
    }
  }
}

#figure(canvas(length: 0.7cm, {
  import draw: *
  let i = 0
  let j = 0
  multiplier(5, 5, size: 1.0)
  set-origin((5, 0))
  multiplier-block((0, 0), 1.0, "so", "co", "pi", "qi", "po", "qo", "si", "ci")
  line("si", (rel:(-0.5, 0.5), to:"si"), mark: (start: "straight"))
  content((rel:(-0.75, 0.75), to:"si"), text(12pt)[$s_i$])
  line("ci", (rel:(0.5, 0), to:"ci"), mark: (start: "straight"))
  content((rel:(0.75, 0), to:"ci"), text(12pt)[$c_i$])
  line("pi", (rel:(0, 0.5), to:"pi"), mark: (start: "straight"))
  content((rel:(0, 0.75), to:"pi"), text(12pt)[$p_i$])
  line("qi", (rel:(0.5, 0), to:"qi"), mark: (start: "straight"))
  content((rel:(0.75, 0), to:"qi"), text(12pt)[$q_i$])
  line("po", (rel:(0, -0.5), to:"po"), mark: (end: "straight"))
  content((rel:(0, -0.75), to:"po"), text(12pt)[$p_i$])
  line("qo", (rel:(-0.5, 0), to:"qo"), mark: (end: "straight"))
  content((rel:(-0.75, 0), to:"qo"), text(12pt)[$q_i$])
  line("so", (rel:(0.5, -0.5), to:"so"), mark: (end: "straight"))
  content((rel:(0.75, -0.75), to:"so"), text(12pt)[$s_o$])
  line("co", (rel:(-0.5, 0), to:"co"), mark: (end: "straight"))
  content((rel:(-0.75, 0), to:"co"), text(12pt)[$c_o$])
  content((5, 0), text(12pt)[$2c_o + s_o = p_i q_i + c_i + s_i$])

  let gate(loc, label, size: 1, name:none) = {
    rect((loc.at(0) - size/2, loc.at(1) - size/2), (loc.at(0) + size/2, loc.at(1) + size/2), stroke: black, fill: white, name: name)
    content(loc, text(11pt)[$label$])
  }
  set-origin((-1.5, -3))
  line((4.5, 0), (-1, 0))  // q
  line((3, 1), (3, -4.5))  // p
  let si = (-1, 1)
  let ci = (4.5, -2.5)
  gate((0.5, -0.5), [$and$], size: 0.5, name: "a1")
  gate((2.5, -0.5), [$and$], size: 0.5, name: "a2")
  gate((2.0, -2.5), [$and$], size: 0.5, name: "a3")
  gate((0.5, -2.5), [$or$], size: 0.5, name: "o1")
  gate((1.5, -1.5), [$xor$], size: 0.5, name: "x1")
  gate((3.5, -3.5), [$xor$], size: 0.5, name: "x2")
  line("a2", (2.5, 0))
  line("x1", (1.5, -0.5))
  line("a2", (3, -0.5))
  line("a2", "a1")
  line("a1", "o1")
  line("a3", "o1")
  line("o1", (rel: (-1.5, 0), to: "o1"))
  line(si, "a1")
  line(ci, "a3")
  line((3.5, -2.5), "x2")
  let turn = (1.5, -3.5)
  line("x1",(rel: (0.5, -2.5), to: si), (rel: (0.5, -0.5), to: si))
  line("x1", turn, "x2")
  line("x2", (rel: (1, -1), to: "x2"))
  line("a3", (2.0, -0.5))
  rect((-0.75, -4), (4, 0.75), stroke: (dash: "dashed"))

  let gate_with_leg(loc, label, size: 1, name:none) = {
    gate(loc, label, size: size, name: name)
    line(name, (rel: (0.5, 0), to: name))
    line(name, (rel: (-0.5, 0), to: name))
    line(name, (rel: (0, 0.5), to: name))
  }
  gate_with_leg((6, 0), [$xor$], size: 0.5, name: "x3")
  content((8, 0), text(12pt)[$= mat(mat(0, 1; 1, 0); mat(1, 0; 0, 1))$])

  gate_with_leg((6, -2), [$or$], size: 0.5, name: "o3")
  content((8, -2), text(12pt)[$= mat(mat(1, 0; 0, 0); mat(0, 1; 1, 1))$])

  gate_with_leg((6, -4), [$and$], size: 0.5, name: "a4")
  content((8, -4), text(12pt)[$= mat(mat(1, 1; 1, 0); mat(0, 0; 0, 1))$])
}), caption: [The array multiplier circuit.])


```julia
using ProblemReductions, GenericTensorNetworks
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
Simulated annealing is an algorithm for finding the global optimum of a given function, which mimics the physical process of cooling down a material. In its simplest form, it is based on the following facts:
- A physical system thermalizes under the Hamiltonian dynamics
- The thermal state at zero temperature is the ground state.
- The thermal distribution are $beta$ and $beta + Delta beta$ are very close when $Delta beta$ is small (@fig:ising-energy-distribution).

#figure(image("../chap4-simulation/images/ising-energy-distribution.svg", width: 70%),
caption: [The binned energy distribution of spin configurations generated unbiasly from the ferromagnetic Ising model ($J_(i j) = -1, L = 10$) at different inverse temperatures $beta$. The method to generate the samples is the tensor network based method detailed in @Roa2024]
) <fig:ising-energy-distribution>

Simulated annealing is a Monte Carlo method that samples the thermal distribution $p_beta$ of a system, with a gradual decrease of temperature. The mixing time of the system is very short when the temperature is high, and the system can be thermalized quickly. As the temperature decreases, the system becomes more likely to be in the low-energy states, and the mixing time becomes longer. We hope the gradual decrease of temperature can make the system thermalize faster given the thermal distribution at the previous temperature.
Simulated annealing is very generic and has been used in various fields. For example, in discrete optimization, it is used to find the global optimum of a given quadratic binary optimization (QUBO) problem @Glover2019:
  $ &min x^T Q x\ &A x = b, quad x in {0, 1}^n $
In quantum circuit simulation, it is used to find the tensor network contraction order @Kalachev2021.
In very-large-scale integration (VLSI) design, it is used to find the optimal layout of transistors @Wong2012.


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
- Logarithmic: $T_k = T_"init"/log(k+1)$ @Geman1984

Theoretically, with a logarithmic cooling schedule that decreases slowly enough, simulated annealing will converge to the global optimum with probability 1. However, in practice, faster cooling schedules are often used with good empirical results.

== Parallel tempering

The energy landscape of a spin glass is rough, with many local minima separated by high barriers. Normal Metropolis sampling is inefficient at low temperatures, as the system gets trapped in a local minimum.
It has been shown that for the spin glass with overlap gap property (right panel) @Gamarnik2021, a local algorithm can not find the global optimum in time faster than exponential time!

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

= Spin dynamics

The dynamics of a Hamiltonian system is governed by the following equation of motion:
$ m (partial^2 bold(x))/(partial t^2) = bold(f)(bold(x)). $

Equivalently, by denoting $bold(v) = (partial bold(x))/(partial t)$, we have the first-order differential equations:
$
cases(m (partial bold(v))/(partial t) &= bold(f)(bold(x)),
(partial bold(x))/(partial t) &= bold(v))
$

It can be solved numerically by various algorithms, including the Verlet algorithm @Verlet1967 and the Euler algorithm.

== The Euler Algorithm

The Euler algorithm is the simplest algorithm for solving the differential equation of motion. 
It is given by:
$
(bold(x)(t+d t), bold(p)(t + d t)) = (bold(x)(t) + (bold(p)(t))/ m d t,  bold(p)(t) + bold(f)(bold(x)(t)) d t)
$
where $bold(p) = m bold(v)$ is the momentum of the particle.

This simple algorithm suffers from the energy drift problem, which is the energy of the system is not conserved. Consider a particle in a harmonic potential $V(x) = 1/2 x^2$, and kinetic energy given by $E_k (p) = 1/2 p^2$.

// Implement the Euler algorithm for a harmonic oscillator
// Returns the position and momentum trajectories
#let euler_oscillator(x0, p0, t, dt) = {
  // Initialize arrays to store trajectories
  let x_traj = (x0,)
  let p_traj = (p0,)
  let e_traj = (0.5 * calc.pow(x0, 2) + 0.5 * calc.pow(p0, 2),)  // Initial energy
  
  // Current state
  let x = x0
  let p = p0
  
  // Number of steps
  let steps = calc.floor(t / dt)
  
  // Simulate the dynamics
  for i in range(steps) {
    // Force for harmonic oscillator: f = -x
    let f = -x
    
    // Update position and momentum using Euler method
    x = x + p * dt
    p = p + f * dt
    
    // Calculate energy
    let energy = 0.5 * calc.pow(x, 2) + 0.5 * calc.pow(p, 2)
    
    // Store the current state
    x_traj.push(x)
    p_traj.push(p)
    e_traj.push(energy)
  }
  
  // Return the trajectories
  return (x_traj, p_traj, e_traj)
}

#let show_trajectory(x_traj, p_traj, e_traj, t_max, t_array) = {
  import draw: *
  let s(it) = text(11pt, it)
  // Set up the plot
  plot.plot(
    size: (6, 4),
    x-label: s[Time],
    y-label: s[$bold(x)$],
    y-min: -2,
    y-max: 2,
    x-min: 0,
    x-max: t_max,
    x-tick-step: 8,
    legend: "legend.north",
    y-tick-step: 1,
    
    // Position trajectory
    plot.add(
      t_array.zip(x_traj),
      label: s[Position],
    ))
  set-origin((8.5, 0))
  plot.plot(
    size: (6, 4),
    x-label: s[Time],
    y-label: s[$E$],
    y-min: -2,
    y-max: 2,
    x-min: 0,
    x-max: t_max,
    x-tick-step: 8,
    legend: "legend.north",
    y-tick-step: 1,
    // Momentum trajectory
    plot.add(
      t_array.zip(e_traj),
      label: s[Energy],
    )
  )
  set-origin((8.5, 0))
  plot.plot(
    size: (6, 4),
    x-label: s[$bold(x)$],
    y-label: s[$bold(p)$],
    y-min: -2,
    y-max: 2,
    x-min: -2,
    x-max: 2,
    x-tick-step: 1,
    legend: "legend.north",
    y-tick-step: 1,
    // Energy trajectory
    plot.add(
      x_traj.zip(p_traj),
      label: s[Phase space],
    )
  )
}

#figure(canvas(length: 0.65cm, {
  import draw: *
  // Set up the canvas dimensions
  let margin = 40
  
  // Run simulation
  let t_max = 20
  let dt = 0.05
  let (x_traj, p_traj, e_traj) = euler_oscillator(1.0, 0.0, t_max, dt)
  
  // Create time array for plotting
  let t_array = range(x_traj.len()).map(i => i * dt)
  show_trajectory(x_traj, p_traj, e_traj, t_max, t_array)
}), caption: [The energy of the system is drifting away from the theoretical value. The parameters are $d t = 0.05$, $t = 20$, initial condition: $bold(x)(0) = 1$, $bold(p)(0) = 0$.]
) <fig:euler-oscillator>

== The Verlet Algorithm

To overcome the issue, the Verlet algorithm is proposed.
The update of position and velocity are separated in time, and is reversible:
#figure(canvas(length: 0.65cm, {
  import draw: *
  let n = 2
  let dx = 6
  let s(it) = text(11pt, it)
  line((0, 0), (2 + (n + 0.5) * dx, 0), mark: (end: "straight"))
  bezier((1, dx/2), (1 + (0.5) * dx, 0), (1.8, dx/2), stroke: (dash: "dashed"), mark: (end: "straight"))
  for (i, (n1, n2)) in (([$bold(v)(t-Delta t)$], [$bold(x)(t-(Delta t)/2)$]), ([$bold(v)(t)$], [$bold(x)(t + (Delta t)/2)$])).enumerate() {
    bezier((1 + i * dx, 0), (1 + (i+1) * dx, 0), (1 + (i+0.5) * dx, dx), mark: (end: "straight"))
    bezier((1 + (i+0.5) * dx, 0), (1 + (i+1.5) * dx, 0), (1 + (i+1) * dx, dx), stroke: (dash: "dashed"), mark: (end: "straight"))
    content((1 + i * dx, -0.5), s[#n1])
    content((1 + (i+0.5) * dx, -0.5), s[#n2])
  }
  bezier((1 + n * dx, 0), (1 + (n + 0.5) * dx, dx/2), (1 + (n + 0.25) * dx, dx/2))
  content((1 + n * dx, -0.5), s[$bold(v)(t + Delta t)$])
  content((1 + (n+0.5) * dx, -0.5), s[$bold(x)(t + 3/2 Delta t)$])
}))


The algorithm is as follows:

#algorithm({
  import algorithmic: *
  Function("Verlet", args: ([$bold(x)$], [$bold(v)$], [$bold(f)$], [$m$], [$d t$], [$n$]), {
    Assign([$bold(v)$], [$bold(v) + (bold(f)(bold(x)))/m d t \/ 2$ #h(2em)#Ic([update velocity at time $d t \/ 2$])])
    For(cond: [$i = 1 dots n$], {
      Cmt[time step $t = i d t$]
      Assign([$bold(x)$], [$bold(x) + bold(v) d t$ #h(2em)#Ic([update position at time $t$])])
      Assign([$bold(v)$], [$bold(v) + (bold(f)(bold(x)))/m d t$ #h(2em)#Ic([update velocity at time $t + d t\/2$])])
    })
    Assign([$bold(v)$], [$bold(v) - (bold(f)(bold(x)))/m d t \/ 2$ #h(2em)#Ic([velocity at time $n d t$])])
    Return[$bold(x)$, $bold(v)$]
  })
})

The Verlet algorithm is a simple yet robust algorithm for solving the differential equation of motion. It is the most widely used algorithm in molecular dynamics simulation.

The most significant advantage of the Verlet algorithm is that it is symplectic, which (approximately) conserves the energy of the system.

#let verlet_oscillator(x0, p0, t, dt) = {
  // Initialize arrays to store trajectories
  let x_traj = (x0,)
  let p_traj = (p0,)
  let e_traj = (0.5 * calc.pow(x0, 2) + 0.5 * calc.pow(p0, 2),)  // Initial energy
  
  // Current state
  let x = x0
  let p = p0
  
  // Number of steps
  let steps = calc.floor(t / dt)
  
  // First half-step update of momentum (Verlet algorithm)
  let f = -x  // Force for harmonic oscillator: f = -x
  p = p + f * dt / 2
  
  // Simulate the dynamics using Verlet algorithm
  for i in range(steps) {
    // Update position
    x = x + p * dt
    
    // Calculate force at new position
    f = -x
    
    // Update momentum (full step)
    p = p + f * dt
    
    // Calculate energy
    let energy = 0.5 * calc.pow(x, 2) + 0.5 * calc.pow(p, 2)
    
    // Store the current state
    x_traj.push(x)
    p_traj.push(p)
    e_traj.push(energy)
  }
  
  // Final half-step adjustment to get velocity at the same time as position
  p = p - f * dt / 2
  // Return the trajectories
  return (x_traj, p_traj, e_traj)
}

#figure(canvas(length: 0.7cm, {
  import draw: *
  let t_max = 20
  let dt = 0.05
  let (x_traj, p_traj, e_traj) = verlet_oscillator(1.0, 0.0, t_max, dt)
  let t_array = range(x_traj.len()).map(i => i * dt)
  show_trajectory(x_traj, p_traj, e_traj, t_max, t_array)
}))

The results are in good agreement with the theoretical values, the energy and position are not drifting away from the theoretical values.

== Simulated bifurcation dynamics

Thermal dynamics can be viewed as the long time limit of the dynamics of the system. Here we use the adiabatic simulated bifurcation (aSB) dynamics@Goto2021 to evolve the spin system to the ground state. Instead of considering boolean spins, we consider the continuous variables $x_i in RR$ that evolve according to the following Hamiltonian:
$
  V_("aSB") = sum_(i=1)^N (x_i^4 / 4 + (a_0 - a(t))/2 x_i^2)- c_0 sum_(i < j) J_(i j) x_i x_j\
  H_("aSB") = a_0/2 sum_(i=1)^N p_i^2 + V_("aSB")\
  dot(p)_i = - (partial V_("aSB"))/(partial x_i)\
  dot(x)_i = (partial H_("aSB"))/(partial p_i)
$
In our implementation, we set the parameters as follows:
- $c_0$ is a constant that tunes the strength of the spin glass energy function. In our case, we set it to $c_0 = 0.5/(sqrt(N) angle.l J angle.r)$, where $angle.l J angle.r = sqrt((sum_(i j) J_(i j)^2)/(N(N-1)))$ is the Frobenius norm of the interaction matrix $J$.
- $a_0$ is set to $1$.

Given the total annealing time $t_"total"$, we slowly drive $a_0$ from $0$ to $2$ with a constant rate $2/t_"total"$. At the initial time, $a(t) = 0$, the ground state of the system is $x_i = 0$ for all $i$. At the final time, $a(t_"total") = 2$, the ground state of the system is the ground state of the spin glass. If the annealing time is long enough, we will find the final state is the ground state of the spin glass.

#figure(image("images/bifurcation_energy_evolution.svg", width: 80%),
caption: [Evolution of energy (left panel) and states (right panel) under different two particle Hamiltonian dynamics (aSB, bSB and dSB) in @Goto2021. $J_12 = J_21 = 1$, times are in units of $0.01$, $c_0 = 0.2$ for all.]
) <fig:bifurcation-energy-evolution>

#bibliography("refs.bib")