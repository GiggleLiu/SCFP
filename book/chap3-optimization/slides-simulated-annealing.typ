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
  title: [Simulated Annealing and Spin Dynamics],
  subtitle: [Physics Inspired Optimization for Spin Glass Ground State Finding],
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

== Challenge resolved: A PR to KrylovKit.jl

#box(stroke: black, inset: 0.5em, [
  Resolve the following issue in KrylovKit.jl: https://github.com/Jutho/KrylovKit.jl/issues/87 . If you can resolve the issue, please submit a pull request to the repository. If your PR is merged, your final grade will be $A+$.
])

Link: https://github.com/Jutho/KrylovKit.jl/pull/125


= Spin glass and computational complexity

== Cooling a system to the ground state
#figure(image("../chap4-simulation/images/ising-energy-distribution.svg", width: 70%),
caption: [The binned energy distribution of spin configurations generated unbiasly from the ferromagnetic Ising model ($J_(i j) = -1, L = 10$) at different inverse temperatures $beta$. The method to generate the samples is the tensor network based method detailed in @Roa2024]
) <fig:ising-energy-distribution>


#align(center, box(stroke: black, inset: 0.5em, [
    Coupling $J_(i j)$ and bias $h_i$ freely tuned $arrow.r$ We get a spin glass!
  ])
)

Spin glass ground state finding problem is hard, it is NP-complete (hardest problems in NP), which is believed to be impossible to solve in polynomial time.

*NP problems*: _Decision problems_, features the property that given a solution, it is _easy to verify_ whether the solution is correct in polynomial time.

== Example: Factoring a number

Solving is hard (foundation of RSA encryption):
#box(text(16pt)[```julia
c = BigInt(21267647932558653302378126310941659999)
@test a * b == c
```])

Verification is easy:
#box(text(16pt)[```julia
a = BigInt(4611686018427387847)
b = BigInt(4611686018427387817)
@test a * b == c
```])

#align(center, box(stroke: black, inset: 0.5em, [
  Easy to verify $!=$ easy to solve (we believe)
]))

== What is an "easy" problem?

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

- $n$ is the size of the input (measured by the number of bits).

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

== Circuit SAT is the hardest problem in NP

=== Example: Encoding the factoring problem to a spin glass
We introduce how to convert the factoring problem into a spin glass problem.
Factoring problem is the cornerstone of modern cryptography, it is the problem of given a number $N$, find two prime numbers $p$ and $q$ such that $N = p times q$.


==
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
    content((rel: (0, 0.3), to: b), text(14pt)[$p_#i$])


    let a2 = "s" + str(i+1) + str(-1) + "'"
    let b2 = (rel: (-0.4, 0.4), to: a2)
    line(a2, b2, mark: (start: "straight"))
    content((rel: (-0.2, 0.2), to: b2), text(14pt)[$0$])

    let a3 = "s" + str(i) + str(n - 1)
    let b3 = (rel: (0.4, -0.4), to: a3)
    line(a3, b3, mark: (end: "straight"))
    content((rel: (0.2, -0.2), to: b3), text(14pt)[$m_#(i+m - 1)$])

  }
  for j in range(n){
    let a = "q0" + str(j)
    let b = (rel: (0.5, 0), to: a)
    line(a, b, mark: (start: "straight"))
    content((rel: (0.3, 0), to: b), text(14pt)[$q_#j$])

    let a2 = "q" + str(m) + str(j) + "'"
    let b2 = (rel: (-0.5, 0), to: a2)
    line(a2, b2, mark: (end: "straight"))


    let a3 = "c" + str(-1) + str(j) + "'"
    let b3 = (rel: (0.5, 0), to: a3)
    line(a3, b3, mark: (start: "straight"))
    content((rel: (0.3, 0), to: b3), text(14pt)[$0$])
  
    if (j < n - 1) {
      let a4 = "c" + str(m - 1) + str(j)
      let b4 = "s" + str(m) + str(j) + "'"
      bezier(a4, b4, (rel: (-1, 0), to: a4), (rel: (-0.5, -1), to: a4), mark: (end: "straight"))
    } else {
      let a4 = "c" + str(m - 1) + str(j)
      line(a4, (rel: (-0.5, 0), to: a4), mark: (end: "straight"))
      content((rel: (-0.8, 0), to: a4), text(14pt)[$m_#(j+m)$])
    }
    if (j < n - 1) {
      let a5 = "s0" + str(j)
      let b5 = (rel: (0.4, -0.4), to: a5)
      line(a5, b5, mark: (end: "straight"))
      content((rel: (0.2, -0.2), to: b5), text(14pt)[$m_#j$])
    }
  }
}

#slide(figure(canvas({
  import draw: *
  let i = 0
  let j = 0
  multiplier(5, 5, size: 1.0)
}), caption: []),
figure(canvas({
  import draw: *
  multiplier-block((0, 0), 1.0, "so", "co", "pi", "qi", "po", "qo", "si", "ci")
  line("si", (rel:(-0.5, 0.5), to:"si"), mark: (start: "straight"))
  content((rel:(-0.75, 0.75), to:"si"), text(14pt)[$s_i$])
  line("ci", (rel:(0.5, 0), to:"ci"), mark: (start: "straight"))
  content((rel:(0.75, 0), to:"ci"), text(14pt)[$c_i$])
  line("pi", (rel:(0, 0.5), to:"pi"), mark: (start: "straight"))
  content((rel:(0, 0.75), to:"pi"), text(14pt)[$p_i$])
  line("qi", (rel:(0.5, 0), to:"qi"), mark: (start: "straight"))
  content((rel:(0.75, 0), to:"qi"), text(14pt)[$q_i$])
  line("po", (rel:(0, -0.5), to:"po"), mark: (end: "straight"))
  content((rel:(0, -0.75), to:"po"), text(14pt)[$p_i$])
  line("qo", (rel:(-0.5, 0), to:"qo"), mark: (end: "straight"))
  content((rel:(-0.75, 0), to:"qo"), text(14pt)[$q_i$])
  line("so", (rel:(0.5, -0.5), to:"so"), mark: (end: "straight"))
  content((rel:(0.75, -0.75), to:"so"), text(14pt)[$s_o$])
  line("co", (rel:(-0.5, 0), to:"co"), mark: (end: "straight"))
  content((rel:(-0.75, 0), to:"co"), text(14pt)[$c_o$])
  content((5, 0), text(14pt)[$2c_o + s_o = p_i q_i + c_i + s_i$])

  let gate(loc, label, size: 1, name:none) = {
    rect((loc.at(0) - size/2, loc.at(1) - size/2), (loc.at(0) + size/2, loc.at(1) + size/2), stroke: black, fill: white, name: name)
    content(loc, text(14pt)[$label$])
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
  content((8, 0), text(14pt)[$= mat(mat(0, 1; 1, 0); mat(1, 0; 0, 1))$])

  gate_with_leg((6, -2), [$or$], size: 0.5, name: "o3")
  content((8, -2), text(14pt)[$= mat(mat(1, 0; 0, 0); mat(0, 1; 1, 1))$])

  gate_with_leg((6, -4), [$and$], size: 0.5, name: "a4")
  content((8, -4), text(14pt)[$= mat(mat(1, 1; 1, 0); mat(0, 0; 0, 1))$])
}), caption: [])
)


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
      (0, -1, "Circuit SAT", yellow),
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

== Live Coding: Julia implementation of the reduction
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

= Simulated annealing
== Simulate the cooling process to find the ground state?

Based on the following facts:
- Fact 1: A physical system thermalizes under the Hamiltonian dynamics
- Fact 2: The thermal state at zero temperature is the ground state.

Used in:
- Finding the quadratic binary optimization (QUBO) problem. @Glover2019
  $ &min x^T Q x\ &A x = b, quad x in {0, 1}^n $
- In quantum circuit simulation, it is used to find the tensor network contraction order. @Kalachev2021
- In very-large-scale integration (VLSI) design, it is used to find the optimal layout of transistors. @Wong2012

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
3. Logarithmic: $T_k = T_"init"/log(k+1)$ @Geman1984

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

  content((dx/2 + 1.5, 1.5), s[Add more _frustration_])
  content((dx/2 + 1.5, 1.1), s[$arrow.double.r$])
}))

= Spin dynamics @Goto2021

== Summarize the dynamics
#timecounter(2)
In the molecular dynamics simulation, we have the following equation of motion:
$ m (partial^2 bold(x))/(partial t^2) = bold(f)(bold(x)). $

Equivalently, by denoting $bold(v) = (partial bold(x))/(partial t)$, we have the first-order differential equations:
$
cases(m (partial bold(v))/(partial t) &= bold(f)(bold(x)),
(partial bold(x))/(partial t) &= bold(v))
$

It is a typical Hamiltonian dynamics, which can be solved numerically by the Verlet algorithm @Verlet1967.


== Dynamics is determined by the energy model
#timecounter(3)

The *force* is the gradient of the energy:
$
bold(f)_i = -nabla_i E(bold(x)_1, bold(x)_2, dots, bold(x)_N)
$
where $nabla_i = (partial_(x_i), partial_(y_i), partial_(z_i))$ is the gradient operator with respect to the $i$-th particle.

The classical dynamics is governed by the *energy model*:
$
E(bold(x)_1, bold(x)_2, dots, bold(x)_N) = sum_(i=1)^N V(bold(x)_i) + sum_(i, j) V(bold(x)_i, bold(x)_j) + sum_(i, j, k) V(bold(x)_i, bold(x)_j, bold(x)_k) + dots
$
where $bold(x)_i$ is the position of the $i$-th particle, $V(bold(x)_i)$ is the potential energy of the $i$-th particle, and $V(bold(x)_i, bold(x)_j, dots)$ is the potential energy of the interaction between the $i$-th, $j$-th particles, and so on.

== Summarize the dynamics
#timecounter(2)
In the molecular dynamics simulation, we have the following equation of motion:
$ m (partial^2 bold(x))/(partial t^2) = bold(f)(bold(x)). $

Equivalently, by denoting $bold(v) = (partial bold(x))/(partial t)$, we have the first-order differential equations:
$
cases(m (partial bold(v))/(partial t) &= bold(f)(bold(x)),
(partial bold(x))/(partial t) &= bold(v))
$

It is a typical Hamiltonian dynamics, which can be solved numerically by the Verlet algorithm @Verlet1967.

== The Euler Algorithm
#timecounter(2)

The Euler algorithm is the simplest algorithm for solving the differential equation of motion. 
It is given by:
$
(bold(x)(t+d t), bold(p)(t + d t)) = (bold(x)(t) + (bold(p)(t))/ m d t,  bold(p)(t) + bold(f)(bold(x)(t)) d t)
$
where $bold(p) = m bold(v)$ is the momentum of the particle.

Q: Will this simple algorithm work?

== Euler Method for Harmonic Oscillator

- Consider a particle in a harmonic potential $V(x) = 1/2 x^2$, and kinetic energy given by $E_k (p) = 1/2 p^2$.
- Parameters: $d t = 0.05$, $t = 20$, initial condition: $bold(x)(0) = 1$, $bold(p)(0) = 0$

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
  // Set up the plot
  plot.plot(
    size: (6, 4),
    x-label: "Time",
    y-label: [$bold(x)$],
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
      label: "Position",
    ))
  set-origin((9, 0))
  plot.plot(
    size: (6, 4),
    x-label: "Time",
    y-label: [$E$],
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
      label: "Energy",
    )
  )
  set-origin((9, 0))
  plot.plot(
    size: (6, 4),
    x-label: [$bold(x)$],
    y-label: [$bold(p)$],
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
      label: "Phase space",
    )
  )
}

#figure(canvas({
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
}))

== The Verlet Algorithm
#timecounter(5)

To overcome the issue, the Verlet algorithm is proposed.
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

== The Verlet Algorithm

#figure(canvas({
  import draw: *
  let n = 2
  let dx = 6
  line((0, 0), (2 + (n + 0.5) * dx, 0), mark: (end: "straight"))
  bezier((1, dx/2), (1 + (0.5) * dx, 0), (1.8, dx/2), stroke: (dash: "dashed"), mark: (end: "straight"))
  for (i, (n1, n2)) in (([$bold(v)(t-Delta t)$], [$bold(x)(t-(Delta t)/2)$]), ([$bold(v)(t)$], [$bold(x)(t + (Delta t)/2)$])).enumerate() {
    bezier((1 + i * dx, 0), (1 + (i+1) * dx, 0), (1 + (i+0.5) * dx, dx), mark: (end: "straight"))
    bezier((1 + (i+0.5) * dx, 0), (1 + (i+1.5) * dx, 0), (1 + (i+1) * dx, dx), stroke: (dash: "dashed"), mark: (end: "straight"))
    content((1 + i * dx, -0.5), text(16pt, n1))
    content((1 + (i+0.5) * dx, -0.5), text(16pt, n2))
  }
  bezier((1 + n * dx, 0), (1 + (n + 0.5) * dx, dx/2), (1 + (n + 0.25) * dx, dx/2))
  content((1 + n * dx, -0.5), text(16pt, [$bold(v)(t + Delta t)$]))
  content((1 + (n+0.5) * dx, -0.5), text(16pt, [$bold(x)(t + 3/2 Delta t)$]))
}))

== The Verlet Algorithm on Harmonic Oscillator

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

#figure(canvas({
  import draw: *
  let t_max = 20
  let dt = 0.05
  let (x_traj, p_traj, e_traj) = verlet_oscillator(1.0, 0.0, t_max, dt)
  let t_array = range(x_traj.len()).map(i => i * dt)
  show_trajectory(x_traj, p_traj, e_traj, t_max, t_array)
}))

The results are in good agreement with the theoretical values, the energy and position are not drifting away from the theoretical values.
It is because the Verlet algorithm is *symplectic*, which conserves the energy of the system.

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