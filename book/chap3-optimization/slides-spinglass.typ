#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
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

Q: What is the configuration of the following spin system? The number on the edges are the couplings $J_(i j)$.
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

For the ferromagnetic Ising model, the ground state is two fold degenerate, they are the all-up and all-down configurations. At zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r = 1$.

To find all degenerate ground states, we can solve the spin glass problem with the `ConfigsMin` solver in #link("https://github.com/QuEraComputing/GenericTensorNetworks.jl")[GenericTensorNetworks.jl].

```julia
julia> using GenericTensorNetworks

julia> solve(spin_glass, ConfigsMin())[]  # solve the spin glass ground state
(-24.0, {0000000000000000, 1111111111111111})ₜ
```
The returned value is a data structure that contains the lowest energy and the associated configurations.
We can see the ground states are two fold degenerate, they are the all-up and all-down configurations.

== Integrate a function with importance sampling

sum == integral

For illustrative purpose, we consider integrating a positive function $f(x)$ defined on a unit square.
$
integral f(x) d x
$
== Importance sampling
Intead of evaluating the integral directly, the importance sampling can sample $x$ with probability $p(x)$ and estimate the integral as the statistical average of $f(x)\/p(x)$.
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

A good sampling probability can improve the efficiency of the estimation. We consider the example in @fig:importance-sampling.
The function is a peak-like function defined on a unit square, which is zero everywhere except a very small region (the red circle of radius $r << 1$) at the origin. The integral of $f(x)$ is 1. If the sampling is uniform, the sampler will spend most of the time sampling the region far away from the origin.
However, if the sample $x$ with $10 times$ more probability near the origin (in the dark blue region), as shown in the right panel, you can have $10 times$ more chance to find a sample in the peak region. So, whenever you find a sample in the dark blue region, you only count it as $0.1$ sample as a compensation. With this small change, the statistical average will be $sqrt(10)$ times more accurate. In a even more extreme case, if the sampled probability $p(x)$ is proportional to $f(x)$, the number of sample to reach exact result is $1$.

== Ergodicity
On the other hand, bad sampling probability $p(x)$ may cause the function $f(x)$ to have a poor estimate even if you have infinite samples. This happens when the *ergodicity* is broken, i.e. the system can not reach the whole configuration space. For example, in @fig:ergodicity, the function $f(x)$ has two peaks, but only one peak is accessible to the sampler. Then, no matter how many samples you have, you can not get a good estimate of the integral.
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


== Metropolis-Hastings algorithm
To evaluate the magnetization, we can sample the configuration space with the Boltzmann distribution $p(bold(s)) = (e^(-beta H(bold(s))))\/Z$, on each sample, we evaluate $|m(bold(s))|$ and calculate the statistical average.
The problem is that we do not know the partition function $Z$. Even if we know how to compute $p(bold(s))$, the sampling is still challenging.
Metropolis-Hastings algorithm is a sampling method to sample the configuration space with un-normalized probability.

The Metropolis-Hastings algorithm is a Markov chain. A Markov chain is a sequence of random variables $bold(s)_1, bold(s)_2, dots$ with the property that the probability of moving to the next state depends only on the current state.
It is characterized by the transition probability $P(bold(s)'|bold(s))$, the probability of moving from $bold(s)$ to $bold(s)'$.

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

The algortihm is summarized as follows:
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

In line 4, a new configuration $bold(s)'$ is proposed. The probability of proposing $bold(s)'$ from $bold(s)$ is $T(bold(s)arrow.r bold(s)')$, which is known when we design the algorithm.
== Example: 3-state Ising model
For example, when we propose a state transfer on two 3-state spins, we may have the following transition rule:
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
The probability of proposing $(1, 0)$ from $(0, 0)$ is $1\/3$, and the probability of proposing $(0, 1)$ from $(0, 0)$ is $1\/2$. This prior is biased towards having less configurations in state $(0, 0)$, we must compensate for this bias by adjusting the acceptance probability. In a two state spin system, the random flip of a spin is unbiased, so the acceptance probability only depends on the ratio of the probability of the new and current configurations:
$ p(bold(s)')/p(bold(s)) = e^(-beta (H(bold(s)') - H(bold(s))). $

== Physical quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$

== Metric of a good MCMC method

=== Acceptance rate
In the ferrromagnetic phase, the MCMC method can easily get stuck in one of the ground states. A clever design can help the sampler to escape the local minimum, the cluster update proposed in @Swendsen1987 is a good example. When the prior is the same as the target distribution, the sampling the the most efficient, it has acceptance rate 1.

=== Autocorrelation time
Because a new sample in the MCMC method is generated from the previous sample, we often have time correlated samples in MCMC methods.
Since the correlated samples are not independent, we effectively have less samples than we expect.
The autocorrelation time $tau$ is the number of steps it takes for the correlation between two consecutive samples to decay to one half of the maximum correlation.
The effective number of independent samples is $n\/tau$. A good MCMC method should have a small autocorrelation time.

= Spin glass

== The hardness
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


== From logic circuit to spin glass
We start by showing the following statement is true: _If you can drive a Spin glass system to the ground state, you can prove any theorem._ Note that theorem proving is not different from other problems in NP. First it is a decision problem, second, with a proof, it is easy to verify whether the solution is correct in polynomial time. A spin glass system can encode a target problem by tuning its couplings $J_(i j)$ and biases $h_i$ in the Hamiltonian.
After driving the spin glass system to the ground state, we can read out the proof from the spin configuration.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
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

The ground state of a spin glass can encode the truth table of any logic circuit. In @tbl:logic-circuit-to-spin-glass, we show the spin glass gadget for well-known logic gates $not$, $and$ and $or$. These logic gates can be used to construct any logic circuit.

#figure(table(columns: 4, table.header([Gate], [Gadget], [Ground states], [Lowest energy]),
[Logical not: $not$], [
  #canvas(length: 0.5cm, {
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
[Logical and: $and$], [#canvas(length: 0.5cm, {
  triangle((1, -2, -2), (1, 1, -2))
})
],
[(-1, -1, -1),\ (+1, -1, +1),\ (-1, +1, +1),\ (+1, +1, +1)], [-3],
[Logical or: $or$], [
#canvas(length: 0.5cm, {
  triangle((1, -2, -2), (-1, -1, 2))
})

],
[(-1, -1, -1),\ (+1, -1, -1),\ (-1, +1, -1),\ (+1, +1, +1)], [-3],
), caption: [The spin glass gadget for logic gates@Gao2024. The blue/red spin is the input/output spins. The numbers on the vertices are the biases $h_i$ of the spins, the numbers on the edges are the couplings $J_(i j)$.]) <tbl:logic-circuit-to-spin-glass>

With these gadgets, we can construct any logic circuits utilizing the composibility of logic gadgets. Given two logic gadgets $H_1$ and $H_2$, in the ground state of the combined gadget $H_1 compose H_2$, both $H_1$ and $H_2$ are in their own ground state. i.e. the logic expressions associated with $H_1$ and $H_2$ are both satisfied. Therefore, the ground state of $H_1 compose H_2$ is the same as the truth table of the composed logic circuit.


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

In this example, the resulting spin glass has $63$ spins, which is beyond the capability of brute force search.
Here we resort to the generic tensor network solver to find the ground state.
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

```julia
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

The spectral gap can be computed as follows:
```julia
using LinearAlgebra: eigvals

function spectral_gap(P)
    eigenvalues = eigvals(P, sortby=x -> real(x))
    return 1.0 - real(eigenvalues[end-1])
end
```

#figure(image("images/spectralgap.svg", width: 300pt),
caption: [Spectral gap v.s. $1\/T$ of the Ising model ($J = -1$) on a circle of length $N=6$.],
)

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



== Parallel tempering

Parallel tempering (also known as replica exchange) is a Monte Carlo method designed to improve sampling efficiency for systems with rough energy landscapes, such as spin glasses. The key idea is to simulate multiple replicas of the system at different temperatures simultaneously, allowing configurations to be exchanged between temperatures.

=== Algorithm overview

In parallel tempering:

1. We simulate $M$ replicas of the system at different temperatures $T_1 < T_2 < ... < T_M$
2. Each replica evolves according to standard Metropolis dynamics at its temperature
3. Periodically, we attempt to swap configurations between adjacent temperature levels

The swap between configurations at temperatures $T_i$ and $T_(i+1)$ is accepted with probability:

$
P_"swap"(bold(s)_i, bold(s)_(i+1)) = min(1, exp(-(beta_i - beta_(i+1))(H(bold(s)_(i+1)) - H(bold(s)_i))))
$

where $beta_i = 1/T_i$ and $bold(s)_i$ is the configuration at temperature $T_i$.

=== Benefits of parallel tempering

Parallel tempering offers several advantages:

1. *Improved exploration*: Higher temperature replicas can easily cross energy barriers, while lower temperature replicas sample the relevant low-energy states
2. *Faster thermalization*: Configurations can travel up and down the temperature ladder, helping the system escape local minima
3. *Better sampling of low-energy states*: The method provides more efficient sampling of the low-temperature distribution

=== Implementation considerations

- *Temperature spacing*: The temperatures should be chosen so that the acceptance rate for swaps between adjacent temperatures is reasonable (typically 20-30%)
- *Swap frequency*: Swaps are typically attempted after each replica has undergone several Metropolis updates
- *Number of replicas*: More replicas provide better temperature coverage but increase computational cost

=== Pseudocode

#algorithm({
  import algorithmic: *
  Function("ParallelTempering", args: ([$H$], [$T_1, ..., T_M$], [$N_"steps"$]), {
    Assign([$bold(s)_1, ..., bold(s)_M$], [random initial configurations])
    For(range: [$t = 1$ to $N_"steps"$], {
      // Update each replica with Metropolis
      For(range: [$i = 1$ to $M$], {
        Assign([$bold(s)_i$], [MetropolisUpdate($bold(s)_i$, $H$, $T_i$)])
      })
      
      // Attempt swaps between adjacent temperatures
      If(cond: [$t$ mod $N_"swap"$ = 0], {
        For(range: [$i = 1$ to $M-1$], {
          Assign([$Delta E$], [$H(bold(s)_(i+1)) - H(bold(s)_i)$])
          Assign([$Delta beta$], [$1/T_i - 1/T_(i+1)$])
          If(cond: [$cal(U)(0,1) < exp(-Delta beta \cdot Delta E)$], {
            Assign([$(bold(s)_i, bold(s)_(i+1))$], [$(bold(s)_(i+1), bold(s)_i)$])
          })
        })
      })
    })
    Return([$bold(s)_1$])  // Return lowest temperature configuration
  })
})

Parallel tempering is particularly effective for spin glass systems where the energy landscape contains many local minima separated by high barriers, making standard Metropolis sampling inefficient at low temperatures.

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

=== Estimating the Cheeger constant

Exactly computing the Cheeger constant is NP-hard, but there are several approaches to estimate it:

1. *Spectral methods*: Using Cheeger's inequality, we can compute $lambda_2$ and use it as an approximation.

2. *Sampling-based methods*: For large graphs, we can use random walks to estimate the conductance.

3. *Approximation algorithms*: There exist polynomial-time algorithms that can approximate the Cheeger constant within certain factors.

For spin glass systems, estimating the Cheeger constant can provide valuable insights into the difficulty of sampling from the Boltzmann distribution at low temperatures. A small Cheeger constant indicates the presence of bottlenecks in the state space, which can significantly slow down the mixing of MCMC methods.

#algorithm({
  import algorithmic: *
  Function("EstimateCheegerConstant", args: ([$G = (V, E)$], [$k$]), {
    // Compute the normalized Laplacian matrix
    Assign([$D$], [diagonal degree matrix of $G$])
    Assign([$A$], [adjacency matrix of $G$])
    Assign([$L$], [$I - D^(-1/2) A D^(-1/2)$])
    
    // Compute the second smallest eigenvalue
    Assign([$lambda_2$], [second smallest eigenvalue of $L$])
    
    // Use spectral partitioning to find a good cut
    Assign([$v_2$], [eigenvector corresponding to $lambda_2$])
    Assign([$S_t$], [vertices with the smallest $t$ values in $D^(-1/2) v_2$])
    
    // Compute conductance for different values of t
    Assign([$h_"min"$], [$infinity$])
    For(range: [$t = 1$ to $|V|-1$], {
      Assign([$h_t$], [$|E(S_t, V backslash S_t)| \/ min("vol"(S_t), "vol"(V backslash S_t))$])
      If(cond: [$h_t < h_"min"$], {
        Assign([$h_"min"$], [$h_t$])
      })
    })
    
    Return([$h_"min"$, $lambda_2$])
  })
})

In practice, for spin glass systems, the Cheeger constant provides a quantitative measure of how "glassy" the energy landscape is. Systems with small Cheeger constants have energy landscapes with high barriers between different metastable states, making equilibration difficult and necessitating techniques like parallel tempering to efficiently sample the state space.

== Example: Ferromagnetic Ising model on Fullerene graph
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


#let fullerene() = {
    let φ = (1+calc.sqrt(5))/2
    let res = ()
    for (x, y, z) in ((0.0, 1.0, 3 * φ), (1.0, 2 + φ, 2 * φ), (φ, 2.0, 2 * φ + 1.0)) {
		    for (α, β, γ) in ((x,y,z), (y,z,x), (z,x,y)) {
			      for loc in ((α,β,γ), (α,β,-γ), (α,-β,γ), (α,-β,-γ), (-α,β,γ), (-α,β,-γ), (-α,-β,γ), (-α,-β,-γ)) {
				        if not res.contains(loc) {
                    res.push(loc)
                }
            }
        }
    }
	  return res
}

#figure(canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(14pt, it)
  let vertices = fullerene()
  let edges = udg-graph(vertices, unit: calc.sqrt(5))
  show-graph(vertices.map(v=>(v.at(0), v.at(1)*calc.sqrt(1/5) + v.at(2)*calc.sqrt(4/5))), edges)
}))


```julia
julia> using GenericTensorNetworks, Graphs, ProblemReductions
julia> function fullerene()  # construct the fullerene graph in 3D space
           th = (1+sqrt(5))/2
           res = NTuple{3,Float64}[]
           for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
               for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
                   for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c), (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                       if loc not in res
                           push!(res, loc)
                       end
                   end
               end
           end
           return res
       end
fullerene (generic function with 1 method)
julia> fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)); # construct the unit disk graph
julia> spin_glass = SpinGlass(fullerene_graph, UnitWeight(ne(fullerene_graph)), zeros(Int, nv(fullerene_graph)));
julia> problem_size(spin_glass)
(num_vertices = 60, num_edges = 90)
julia> log(solve(spin_glass, PartitionFunction(1.0))[])/nv(fullerene_graph)
1.3073684577607942
julia> solve(spin_glass, CountingMin())[]
(-66.0, 16000.0)ₜ
```

= Hands-on
== Hands-on: Implement and improve a simple Lanczos algorithm
1. Run the demo code in folder: `SpinGlass/examples` with:
   ```bash
   $ make init-SpinGlass
   $ make example-SpinGlass
   ```

==
#bibliography("refs.bib")