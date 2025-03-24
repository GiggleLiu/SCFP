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
  title: [Spin Systems and Monte Carlo Methods],
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


#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
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

= Spin Systems

== What is a spin?

A spin is a $plus.minus 1$ variable, with physical interpretation of a magnetic moment.
#figure(canvas({
  import draw: *
  let s(it) = text(18pt, it)
  line((0, 0), (0, 1.2), stroke: (paint: red, thickness: 4pt), mark: (end: "straight"))
  circle((0, 0.5), radius: 0.2, fill: black, stroke: none)
  content((1, 0.5), s[+1])

  let dx = 3
  line((dx, -0.2), (dx, 1.0), stroke: (paint: blue, thickness: 4pt), mark: (start: "straight"))
  circle((dx, 0.5), radius: 0.2, fill: black, stroke: none)
  content((dx + 1, 0.5), s[-1])
}))
In this course, we treat a spin as a logical variable, $+1$ represents false, $-1$ represents true.

When two parallel aligned spins have lower energy, they attract each other, the interaction is called *ferromagnetic*. Otherwise, they repel each other, the interaction is called *antiferromagnetic*.

=== References
- Lecture note of Anders Sandvik: #link("https://physics.bu.edu/~py502/lectures5/mc.pdf")[Monte Carlo simulations in classical statistical physics]
- Book: The nature of computation, @Moore2011, Chapter 12-13

== Ising model

An Ising model is defined by a graph $G = (V, E)$ and a set of spins $s_i in {-1, 1}$ for each vertex $i in V$. The energy of the system is given by

$
H = sum_((i,j) in E) J_(i j) s_i s_j + sum_(i in V) h_i s_i
$ <eq:spin-glass-hamiltonian>

Q: What is the configuration with the lowest energy in the following spin system? The number on the edges are the couplings $J_(i j)$.
#figure(canvas({
  import draw: *
  triangle((-1, -1, -1), (none, none, none), colors: (white, white, white))
  set-origin((7, 0))
  triangle((1, 1, 1), (none, none, none), colors: (white, white, white))
}))


== Defining an Ising model

Through the package #link("https://github.com/GiggleLiu/ProblemReductions.jl")[ProblemReductions.jl], the `SpinGlass` type can be constructed from a graph and a set of couplings and fields.

```julia
julia> using ProblemReductions, Graphs
julia> grid_graph = grid((4, 4));
julia> spin_glass = SpinGlass(
           grid_graph,   # graph
           -ones(Int, ne(grid_graph)),    # J, in order of edges
           zeros(Int, nv(grid_graph))     # h, in order of vertices
       );
julia> energy(spin_glass, -ones(16))  # energy of the all-down configuration
-24
```

== Phase transition

#figure(image("images/magnets.png", width: 200pt))

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


#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  spinconfig(4, 4, random_bools(0.5).slice(4, 20), x: 2)
  spinconfig(4, 4, random_bools(0.0).slice(4, 20), x: -6)
  line((-5, -1), (0, -1), stroke: (paint: blue, thickness: 2pt), mark: (start: "straight"))
  line((0, -1), (5, -1), stroke: (paint: red, thickness: 2pt), mark: (end: "straight"))
  content((0, -1.5), s[Temperature $T = 1\/beta$])
  content((0, 0), s[$T_c$], name: "T_c")
  line((0, -0.4), (0, -1), mark: (end: "straight"))
  content((6, -1), s[Hot])
  content((-6, -1), s[Cold])
  content((-4.5, 1.5), s[$angle.l |m| angle.r != 0$])
  content((3.5, 1.5), s[$angle.l |m| angle.r = 0$])
})) <fig:phase-transition>


== Ferromagnetic Ising model

Setup:
- $J_(i j) = -1$ (ferromagnetic)
- $L times L$ grid (no-frustration)

#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  spinconfig(4, 4, random_bools(0.5).slice(4, 20))
  content((5, 1.5), s[$|S| = 2^(L^2)$])
}))
The solution space as $S$, and $bold(s) in S$ is the configuration of spins, e.g. $bold(s) = {-1, -1, 1, -1, 1, dots}$. The number of configurations is $|S| = 2^(L^2)$.

== Probabilistic description
System can be described by a probability distribution, called the Boltzmann distribution:
$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,
$
where $beta = 1 \/ k_B T$ is the inverse temperature, $Z$ is the partition function $Z = sum_bold(s) e^(-beta H(bold(s)))$ that normalizes the probability distribution.

- _Remark_: Configurations with lower energy have higher probability to be observed.
- _Remark_: The sensitivity of the probability distribution to the energy is determined by the inverse temperature $beta$. At zero temperature, $beta$ is infinite, and only the ground state can be observed.

== Order parameter
*Definition (Order parameter)*: _An observable that identifies a phase._

Order parameter for the magnetization
$ |m| = lr(|sum_(i=1)^(L^2) s_i\/L^2|). $ <eq:magnetization>
The statistical average
$
angle.l |m| angle.r = sum_(bold(s) in S) |m(bold(s))| p(bold(s))
$ <eq:magnetization-average>
At the inifinite size limit ($L arrow.r infinity$), if the statistical average $angle.l |m| angle.r$ is non-zero, the system is in the magnetized phase. If $angle.l |m| angle.r$ is zero, the system is in the disordered phase.

== Nature favors ground states
For the ferromagnetic Ising model, the ground state is two fold degenerate, they are the all-up and all-down configurations. At zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r = 1$.

To find all degenerate ground states, we can solve the spin glass problem with the `ConfigsMin` solver in #link("https://github.com/QuEraComputing/GenericTensorNetworks.jl")[GenericTensorNetworks.jl].

#box(text(16pt, [```julia
julia> using GenericTensorNetworks

julia> solve(spin_glass, ConfigsMin())[]  # solve the spin glass ground state
(-24.0, {0000000000000000, 1111111111111111})ₜ
```
]))

== Estimate the observables at finite temperature

$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,\
angle.l O angle.r = sum_(bold(s) in S) O(bold(s)) p(bold(s))
$

=== A statistical approach
1. Draw sample from $p(bold(s))$,
2. Take the statistical average of $O(bold(s))$.

== Importance sampling
Consider integrating a *positive* function $f(x)$ defined on a unit square.
$
integral f(x) d x
$
Instead of randomly draw samples (left), the importance sampling can sample $x$ with probability $p(x)$ and estimate the integral as the statistical average of $f(x)\/p(x)$.
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(80%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  content((0, 0.5), s[$f(x) = cases(1/(pi r^2)\, &|x| < r, 0\, &"otherwise")$])
  content((0, -2.5), s[Uniform sampling])

  set-origin((6, 0))
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(90%))
  rect((-0.5, -0.5), (0.5, 0.5), stroke: none, fill: blue.lighten(50%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  content((0, -0.8), s[$10 times$])
  content((0, -2.5), s[Importance sampling])
})) <fig:importance-sampling>

== Risk: Lost of ergodicity
*Ergodicity*: Every configuration in the configuration space is reachable from any other configuration in a finite number of steps.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(100%))
  rect((-1.5, -1.5), (0.5, 0.5), stroke: none, fill: blue.lighten(50%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  circle((0.8, 0.8), radius: 0.1, fill: red, stroke: none)
  content((0, -2.5), s[Ergodicity is broken])
  content((-0.5, -0.7), s[Sample\ region])
})) <fig:ergodicity>

Q: If $p(x)$ is proportional to $f(x)$, what is the variance of the estimate?

== The remaining problem

Now you know why sampling from $p(bold(s))$ is good?
$
angle.l O angle.r = sum_(bold(s) in S) O(bold(s)) p(bold(s))
$

- The variance in the order of magnitude of $p(bold(s))$ is much larger than that of $O(bold(s))$.

=== Issues
1. How to compute $p(bold(s)) = e^(-beta H(bold(s)))\/Z$ (we do not know $Z$)?
2. Even if we know how to compute $p(bold(s))$, how to draw sample from $p(bold(s))$?


// Use a tensor network to characterize the probability model, and then use Monte Carlo method to sample the model.

== Markov chain Monte Carlo
- Task: Given a *unnormalized* probability distribution $p(bold(s))$, generate a sequence of samples $bold(s)_1, bold(s)_2, dots$ that converges to the true distribution.
- Markovian: The probability of moving to the next state $bold(s)_(k+1)$ depends only on the current state $bold(s)_k$. Characterized by the transition probability $P(bold(s)'|bold(s))$.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  let dx = 3
  for i in range(5) {
    content((i * dx, 0), boxed(s[$bold(s)_#i$]), name: "s" + str(i))
  }
  for i in range(4) {
    line("s" + str(i), "s" + str(i + 1), stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "t" + str(i))
    content((rel: (0, 0.5), to: "t" + str(i) + ".mid"), s[$P(bold(s)_#(i+1)|bold(s)_#i)$])
  }
    
})) <fig:markov-chain>

=== A sufficient condition for the convergence
1. Detailed balance: $p(bold(s)) P(bold(s)'|bold(s)) = p(bold(s)') P(bold(s)|bold(s)')$
2. Ergodicity: Every configuration in the configuration space is reachable from any other configuration in a finite number of steps.

== Metropolis-Hastings algorithm
An MCMC algorithm that satisfies the above two conditions:

#algorithm({
  import algorithmic: *
  Function([Metropolis-Hastings], args: ([$beta$], [$n$]), {
    Assign([$bold(s)$], [Initial a random configuration])
    For(cond: [$i = 1$ to $n$], {
      Assign([$bold(s)'$], [propose a new configuration with prior probability $T(bold(s)arrow.r bold(s)')$])
      Assign([$bold(s)$], [$bold(s)' "with probability:" min(1, (T(bold(s)' arrow.r bold(s)) p(bold(s)'))/(T(bold(s) arrow.r bold(s)') p(bold(s))))$])
    })
  })
 })

In our case, $p(bold(s)')/p(bold(s)) = e^(-beta (H(bold(s)') - H(bold(s))).$ The prior transition probability $T(bold(s)arrow.r bold(s)')$ is the probability of proposing $bold(s)'$ from $bold(s)$.

== Example: 3-state Ising model
The transition rule of a 2-spin model:
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let d = 1.5
  for i in range(2){
    for j in range(2){
      rect((i * d, j * d), (i * d + d, j * d + d))
    }
  }
  line((0, 0), (0, 0.5 * d), mark: (end: "straight"), stroke: (paint: blue, thickness: 2pt))
  line((0, 0), (0.5 * d, 0), mark: (end: "straight"), stroke: (paint: blue, thickness: 2pt))
  content((0, -0.5), s[$(0, 0)$])
  content((d, -0.5), s[$(1, 0)$])

  for target in ((1.5 * d, 0), (1 * d, 0.5 * d), (0.5 * d, 0)) {
    line((1 * d, 0), target, mark: (end: "straight"), stroke: (paint: red, thickness: 2pt))
  }
})) <fig:propose-new-config>
- $T((1, 0) arrow.r (0, 0)) = 1\/3$
- $T((0, 1) arrow.r (0, 0)) = 1\/2$

The bias should be compensated by the acceptance probability.

== Physical quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$


== Results


== Metric of a good MCMC method

=== Acceptance rate
In the ferrromagnetic phase, the MCMC method can easily get stuck in one of the ground states. A clever design can help the sampler to escape the local minimum, the cluster update proposed in @Swendsen1987 is a good example. When the prior is the same as the target distribution, the sampling the the most efficient, it has acceptance rate 1.

=== Autocorrelation time
Because a new sample in the MCMC method is generated from the previous sample, we often have time correlated samples in MCMC methods.
Since the correlated samples are not independent, we effectively have less samples than we expect.
The *autocorrelation time $tau$* is the number of steps it takes for the correlation between two consecutive samples to decay to one half of the maximum correlation.
The effective number of independent samples is $n\/tau$. A good MCMC method should have a small autocorrelation time.

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

== Composibility of logic gadgets
TODO: draw a figure

With these gadgets, we can construct any logic circuits utilizing the composibility of logic gadgets. Given two logic gadgets $H_1$ and $H_2$, in the ground state of the combined gadget $H_1 compose H_2$, both $H_1$ and $H_2$ are in their own ground state. i.e. the logic expressions associated with $H_1$ and $H_2$ are both satisfied. Therefore, the ground state of $H_1 compose H_2$ is the same as the truth table of the composed logic circuit.


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

A cooling process is characterized by lowering the temperature $T$ from a high initial temperature $T_"init"$ to a low final temperature $T_"final"$ following a cooling schedule. The temperature determining the probability distribution of states through the Boltzmann statistics:
$
  p(bold(s)) ~ e^(-H(bold(s))\/T)
$
At the thermal equilibrium, the system effectively finds the distribution with the lowest free energy:
$F = angle.l H angle.r - T S$, where $S$ is the entropy. When the temperature $T$ is high, the system tends to find the distribution with large entropy, making the system behave more randomly. As the temperature decreases, the system tends to find the distribution with lower energy, making the system more likely to be in the low-energy states. This transition can not happen abruptly, otherwise the system will get stuck in a local minimum. We have to wait the dynamics to thermalize the system.

==
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

== Example: Encoding the factoring problem to a spin glass
We introduce how to convert the factoring problem into a spin glass problem.
Factoring problem is the cornerstone of modern cryptography, it is the problem of given a number $N$, find two prime numbers $p$ and $q$ such that $N = p times q$.

== The spectral gap

The transition matrix $P$ of a Markov chain has eigenvalues $1 = lambda_1 > lambda_2 >= lambda_3 >= ... >= lambda_n >= -1$. The spectral gap is defined as:

$
Delta = 1 - lambda_2
$

This gap determines how quickly the Markov chain converges to its stationary distribution (the Boltzmann distribution in our case). A larger spectral gap means faster convergence:

- If $Delta$ is large, the system thermalizes quickly
- If $Delta$ is small, the system thermalizes slowly
- If $Delta approx 0$, the system may never thermalize in practical time

For a Metropolis-Hastings algorithm sampling from the Boltzmann distribution, the mixing time (time to reach equilibrium) scales as $t_"mix" ~ 1/Delta$.

=== Example: Spectral gap and mixing time

Let us consider an Ising model with $N$ spins on a circle:
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let N = 6
  let d = 1.0
  for i in range(N) {
    circle((i * d, 0), radius: 0.2, name: "s" + str(i))
    if i > 0 {
      line("s" + str(i - 1), "s" + str(i))
    }
  }
  bezier("s0.south", "s" + str(N - 1) + ".south", (0, - 1), (N * d - d, -1))

  // Draw a 3D cube to represent the state space
  let cube_center = (8, 0)
  let size = 1.2
  draw_cube(cube_center, size, perspective: 0.3)
  circle("f0", radius: 0.05, fill: red, stroke: none, name: "a")
  circle("f1", radius: 0.05, fill: blue, stroke: none, name: "b")
  bezier("a.east", "b.east", (8, -1), mark: (end: "straight"), stroke: (dash: "dashed"))
  // Add label
  content((cube_center.at(0), cube_center.at(1) - size - 0.3), s[State space for $N=3$])
}))

$ H(bold(s)) = J sum_(i=1)^(N) s_i s_(i+1), quad s_1 = s_(N+1). $


The state space is a hypercube, the number of states is $2^N$. For a given state $bold(s)$, it has $N$ directions to move to the adjacent states $bold(s)', |bold(s') - bold(s)| = 1$, where the norm is the Hamming distance. There is no bias in the prior, i.e. $T(bold(s) arrow.r bold(s)') = T(bold(s)' arrow.r bold(s)) = 1\/N$. So the transition probability from $bold(s)$ to $bold(s')$ is
$ P(bold(s)'|bold(s)) = 1/N min(1, e^(-beta (H(bold(s)') - H(bold(s))))). $

Let's examine how the spectral gap affects mixing time with a concrete example. We'll create a spin glass system and analyze its spectral properties using Julia:

#box(text(16pt)[```julia
using ProblemReductions, Graphs, Printf

function transition_matrix(model::SpinGlass, beta::T) where T
    N = num_variables(model)
    P = zeros(T, 2^N, 2^N)  # P[i, j] = probability of transitioning from j to i
    readbit(cfg, i::Int) = (cfg >> (i - 1)) & 1  # read the i-th bit of cfg
    int2cfg(cfg::Int) = [readbit(cfg, i) for i in 1:N]
    for j in 1:2^N
        for i in 1:2^N
            if count_ones((i-1) ⊻ (j-1)) == 1  # Hamming distance is 1
                P[i, j] = 1/N * min(one(T), exp(-beta * (energy(model, int2cfg(i-1)) - energy(model, int2cfg(j-1)))))
            end
        end
        P[j, j] = 1 - sum(P[:, j])  # rejected transitions
    end
    return P
end
```
])

The spectral gap can be computed as follows:

#box(text(16pt)[```julia
using LinearAlgebra: eigvals

function spectral_gap(P)
    eigenvalues = eigvals(P, sortby=x -> real(x))
    return 1.0 - real(eigenvalues[end-1])
end
```
])

== Spectral gap and mixing time
Spectral gap v.s. $1\/T$ of the Ising model ($J = -1$) on a circle of length $N=6$.
#figure(image("images/spectralgap.svg", width: 400pt))

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

== Cheeger's inequality

Cheeger's inequality is a fundamental result in spectral graph theory that relates the conductance (or isoperimetric constant) of a graph to its spectral gap. This relationship is particularly important in the context of spin glass systems and Markov Chain Monte Carlo methods, as it provides bounds on mixing times.

=== Conductance and the Cheeger constant

For a graph $G = (V, E)$ with vertex set $V$ and edge set $E$, the Cheeger constant (or conductance) $h(G)$ is defined as:

$
h(G) = min_(S subset V, 0 < |S| <= |V|/2) frac(|E(S, V backslash S)|, min("vol"(S), "vol"(V backslash S)))
$

where:
- $E(S, V backslash S)$ is the set of edges between $S$ and its complement
- $"vol"(S) = sum_(v in S) d_v$ is the volume of set $S$, with $d_v$ being the degree of vertex $v$

The Cheeger constant measures how well-connected the graph is, or equivalently, how difficult it is to partition the graph into disconnected components.

=== Cheeger's inequality

Cheeger's inequality relates the Cheeger constant $h(G)$ to the second smallest eigenvalue $lambda_2$ of the normalized Laplacian matrix $L = I - D^(-1/2) A D^(-1/2)$, where $D$ is the degree matrix and $A$ is the adjacency matrix:

$
frac(lambda_2, 2) <= h(G) <= sqrt(2 lambda_2)
$

This inequality provides both lower and upper bounds on the Cheeger constant in terms of the spectral gap.

=== Relation to mixing time

The mixing time of a Markov chain is the time required for the chain to approach its stationary distribution. For a reversible Markov chain, the mixing time $t_"mix"$ is related to the spectral gap $(1 - lambda_2)$ of the transition matrix:

$
t_"mix" approx frac(1, 1 - lambda_2)
$

By Cheeger's inequality, we know that:

$
1 - lambda_2 >= frac(h(G)^2, 2)
$

Therefore:

$
t_"mix" <= frac(2, h(G)^2)
$

This means that a graph with a large Cheeger constant (good expansion properties) will have a small mixing time, allowing MCMC methods to converge quickly to the stationary distribution.

= Hands-on
== Hands-on: Implement and improve a simple Lanczos algorithm
1. Run the demo code in folder: `SpinGlass/examples` with:
   ```bash
   $ make init-SpinGlass
   $ make example-SpinGlass
   ```

==
#bibliography("refs.bib")