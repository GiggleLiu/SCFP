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
  title: [Ising Model and Monte Carlo Methods],
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

= Spin Systems and Phase Transitions

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
The solution space as $S$, and $bold(s) in S$ is the configuration of spins, e.g. $bold(s) = {-1, -1, 1, -1, 1, dots}$.

== Phase transition

#figure(image("images/magnets.png", width: 200pt))
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

== Order parameter
*Definition (Order parameter)*: _An observable that identifies a phase._

Order parameter for the magnetization
$ |m| = lr(|sum_(i=1)^(L^2) s_i\/L^2|). $ <eq:magnetization>
The statistical average
$
angle.l |m| angle.r = sum_(bold(s) in S) |m(bold(s))| p(bold(s))
$ <eq:magnetization-average>
At the inifinite size limit ($L arrow.r infinity$), if the statistical average $angle.l |m| angle.r$ is non-zero, the system is in the magnetized phase. If $angle.l |m| angle.r$ is zero, the system is in the disordered phase.


== Probabilistic description
System can be described by a probability distribution, called the Boltzmann distribution:
$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,
$
where $beta = 1 \/ k_B T$ is the inverse temperature, $Z$ is the partition function $Z = sum_bold(s) e^(-beta H(bold(s)))$ that normalizes the probability distribution.

- _Remark_: Configurations with lower energy have higher probability to be observed.
- _Remark_: The sensitivity of the probability distribution to the energy is determined by the inverse temperature $beta$. At zero temperature, $beta$ is infinite, and only the ground state can be observed.

== Defining an Ising model

Through the package #link("https://github.com/GiggleLiu/ProblemReductions.jl")[ProblemReductions.jl], the `SpinGlass` type can be constructed from a graph and a set of couplings and fields.

#box(text(16pt)[```julia
julia> using ProblemReductions, Graphs

julia> grid_graph = grid((4, 4));
julia> spin_glass = SpinGlass(
           grid_graph,   # graph
           -ones(Int, ne(grid_graph)),    # J, in order of edges
           zeros(Int, nv(grid_graph))     # h, in order of vertices
       );

julia> energy(spin_glass, ones(16))  # NOTE: 0 -> +1, 1 -> -1 in the package
-24
```
])

== Nature favors ground states
For the ferromagnetic Ising model, the ground state is two fold degenerate, they are the all-up and all-down configurations. At zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r = 1$.

To find all degenerate ground states, we can solve the spin glass problem with the `ConfigsMin` solver in #link("https://github.com/QuEraComputing/GenericTensorNetworks.jl")[GenericTensorNetworks.jl].

#box(text(16pt, [```julia
julia> using GenericTensorNetworks

julia> solve(spin_glass, ConfigsMin())[]  # solve the spin glass ground state
(-24.0, {0000000000000000, 1111111111111111})ₜ
```
]))

== The low energy landscape

#figure(image("images/grid66.svg", width: 70%))

- The two degenerate ground states are the all-up and all-down configurations.
- Configurations are connected if they differ by flipping a single spin.

== Estimate the observables at finite temperature

$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,\
angle.l O angle.r = sum_(bold(s) in S) O(bold(s)) p(bold(s))
$

=== A statistical approach
1. Draw sample from $p(bold(s))$,
2. Take the statistical average of $O(bold(s))$.

=== Issues
- The solution space is exponentially large
- We do not know the partition function $Z$

== Monte Carlo method
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

== Results: Simple update MCMC
- $T_c approx 2.269$, below this temperature, the system is in the ferromagnetic phase.
#grid(columns: 2, column-gutter: 50pt, figure(image("images/ising-spins-1.0.gif", width: 300pt)), figure(image("images/ising-spins-3.0.gif", width: 300pt)), align(center)[$T = 1$], align(center)[$T = 3$])

== Quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$

- Autocorrelation function:
$
  A_Q (tau) = (angle.l Q_k Q_(k+tau) angle.r - angle.l Q_k angle.r^2)/(angle.l Q_k^2 angle.r - angle.l Q_k angle.r^2).
$


#figure(image("images/ising-data.png", width: 80%))


== Results: Cluster update MCMC
- Swendsen-Wang update: Instead of updating one spin at a time, we update a cluster of spins at a time.
#grid(columns: 2, column-gutter: 50pt, figure(image("images/swising-spins-1.0.gif", width: 300pt)), figure(image("images/swising-spins-3.0.gif", width: 300pt)), align(center)[$T = 1$], align(center)[$T = 3$])


==
#figure(image("images/sw-data.png", width: 80%))

== Metric of a good MCMC method

=== Acceptance rate
In the ferrromagnetic phase, the MCMC method can easily get stuck in one of the ground states. A clever design can help the sampler to escape the local minimum, the cluster update proposed in @Swendsen1987 is a good example. When the prior is the same as the target distribution, the sampling the the most efficient, it has acceptance rate 1.

=== Autocorrelation time
Because a new sample in the MCMC method is generated from the previous sample, we often have time correlated samples in MCMC methods.
Since the correlated samples are not independent, we effectively have less samples than we expect.
The *autocorrelation time $tau$* is the number of steps it takes for the correlation between two consecutive samples to decay to one half of the maximum correlation.
The effective number of independent samples is $n\/tau$. A good MCMC method should have a small autocorrelation time.

= Spectral gap and mixing time
== Example: Ising model on a circle

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


For a given state $bold(s)$, it has $N$ directions to move to the adjacent states $bold(s)', |bold(s') - bold(s)| = 1$, where the norm is the Hamming distance.
So the transition probability from $bold(s)$ to $bold(s')$ is
$ P(bold(s)'|bold(s)) = 1/N min(1, e^(-beta (H(bold(s)') - H(bold(s))))). $


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


== Live coding: Computing the transition matrix

Spectral gap v.s. $1\/T$ of the Ising model ($J = -1$) on a circle of length $N=6$.
#figure(image("images/spectralgap.svg", width: 400pt))

== Estimate the gap: Cheeger's inequality

Cheeger's inequality is a fundamental result in spectral graph theory that relates the *conductance* of a graph to its *spectral gap*.

For a graph $G = (V, E)$ with vertex set $V$ and edge set $E$, the *Cheeger constant* (or conductance) $h(G)$ is defined as the probability of escaping from the most inescapable set:

$
h(G) = min_(S subset V, 0 < |S| <= |V|/2) frac(|E(S, V backslash S)|, min("vol"(S), "vol"(V backslash S)))
$

where:
- $E(S, V backslash S)$ is the set of edges between $S$ and its complement
- $"vol"(S) = sum_(v in S) d_v$ is the volume of set $S$, with $d_v$ being the degree of vertex $v$

== Cheeger's inequality - Intuition


#figure(image("images/grid66.svg", width: 70%))

== Cheeger's inequality
Cheeger's inequality relates the Cheeger constant $h(G)$ to the second smallest eigenvalue $lambda_2$ of the normalized Laplacian matrix $L = I - D^(-1/2) A D^(-1/2)$, where $D$ is the degree matrix and $A$ is the adjacency matrix:

$
frac(lambda_2, 2) <= h(G) <= sqrt(2 lambda_2)
$

This inequality provides both lower and upper bounds on the Cheeger constant in terms of the spectral gap.

== Relation to mixing time

For a reversible Markov chain, the mixing time $t_"mix"$ is related to the spectral gap $(1 - lambda_2)$ of the transition matrix:

$
t_"mix" approx frac(1, 1 - lambda_2)
$

By Cheeger's inequality, we know that:

$
1 - lambda_2 >= frac(h(G)^2, 2)
arrow.double.r
t_"mix" <= frac(2, h(G)^2)
$

This means that a graph with a large Cheeger constant (good expansion properties) will have a small mixing time, allowing MCMC methods to converge quickly to the stationary distribution.


= Hands-on
== Hands-on: Implement and improve a simple Lanczos algorithm
1. Run the demo code in folder: `IsingModel/examples` with:
   ```bash
   $ make init-IsingModel
   $ make example-IsingModel
   ```
2. Read the code in `IsingModel/src/ising2d.jl`, change the ferromagenetic coupling to antiferromagnetic, and run the simulation again.
3. Read the code in `IsingModel/src/ising2d.jl`, change the ferromagenetic coupling to random coupling, and run the simulation again.

==
#bibliography("refs.bib")