#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.4.1": canvas, draw, tree, vector, decorations
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:1.0.3"
#import algorithmic: algorithm

#show: book-page.with(title: "Monte Carlo methods")
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

#align(center, [= Monte Carlo methods\
_Jin-Guo Liu_])


== Main references
- Lecture note of Anders Sandvik: #link("https://physics.bu.edu/~py502/lectures5/mc.pdf")[Monte Carlo simulations in classical statistical physics]
- Book: The nature of computation, @Moore2011, Chapter 12-13
- Code: https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/IsingModel and https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/Spinglass

== Ising model

An Ising model is defined by a graph $G = (V, E)$ and a set of spins $s_i in {-1, +1}$ for each vertex $i in V$, where $-1$ and $+1$ represent spin-down and spin-up, respectively. The energy of the system is given by
$
H(bold(s)) = sum_((i,j) in E) J_(i j) s_i s_j + sum_(i in V) h_i s_i,
$ <eq:spin-glass-hamiltonian>
where $J_(i j)$ is the interaction strength between spins $i$ and $j$, and $h_i$ is the external field at vertex $i$. $bold(s)$ is the configuration of spins, e.g. $bold(s) = {-1, -1, 1, -1, 1, dots}$. The solution space of an Ising model is the set of all possible configurations of spins, which is exponentially large: $|S| = 2^(|V|)$.

Ising models is the simplest model for us to understand the phenomemon of phase transition, i.e. how magnetization emerges from the disorder. It is also an interesting model to study the computational complexity theory. Finding its ground state (the configuration with the lowest energy) is in general hard, which is known as the spin glass problem. It is in complexity class NP-complete@Moore2011. To construct an Ising model in Julia, we can use the `SpinGlass` function in the #link("https://github.com/GiggleLiu/ProblemReductions.jl")[`ProblemReductions`] package:

```julia
julia> using ProblemReductions, Graphs

julia> grid_graph = grid((4, 4))

julia> spin_glass = SpinGlass(
           grid_graph,   # graph
           -ones(Int, ne(grid_graph)),     # J, in order of edges
           zeros(Int, nv(grid_graph))     # h, in order of vertices
       );

julia> energy(spin_glass, ones(Int, 16))  # energy of the all-down configuration
-24
```

== Phase transition

In this section, we consider the ferromagnetic Ising model ($J_(i j) = -1$) on the $L times L$ grid.
We choose a two dimensional grid because 1D Ising model does not exhibit a phase transition. In 2D, the ferromagnetic Ising model undergoes a phase transition at a critical temperature $T_c$.
Above $T_c$, the system is in the disordered phase, where the spins are randomly oriented. Below $T_c$, the system is in the magnetized phase, where the spins are aligned in the same direction.

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

// #figure(canvas({
//   import draw: *
//   let s(it) = text(14pt, it)
//   spinconfig(4, 4, random_bools(0.5).slice(4, 20))
//   content((5, 1.5), s[$|S| = 2^(L^2)$])
// }))

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

Computation is a valid strategy to study the phase transition.
At finite temperature, the system can be described by a probability distribution, called the Boltzmann distribution:
$
p_beta (bold(s)) = (e^(-beta H(bold(s))))/Z,
$
where $beta = 1 \/ k_B T$ is the inverse temperature, $Z$ is the partition function $Z = sum_bold(s) e^(-beta H(bold(s)))$ that normalizes the probability distribution.
Configurations with lower energy have higher probability to be observed. The sensitivity of the probability distribution to the energy is determined by the inverse temperature $beta$. At zero temperature, $beta$ is infinite, and only the ground state can be observed. However, directly storing this probability distribution is infeasible in a computer program, since it requires exponentially large memory.
The tensor network based method@Roa2024 allows us to peek the spin configurations from the Boltzmann distribution:
```julia
using TensorInference

graph = grid((10, 10))
problem = SpinGlass(graph, -ones(Int, ne(graph)), zeros(Int, nv(graph)))

β = 1.0
pmodel = TensorNetworkModel(problem, β)
samples = sample(pmodel, 1000)
energy_distribution = energy.(Ref(problem), samples)
```
The computation is done by #link("https://github.com/TensorBFS/TensorInference.jl/")[TensorInference.jl]@Roa2024.
We first construct a tensor network model at given inverse temperature $beta$ from a ferromagnetic Ising model on a $10 times 10$ grid, and then sample the model to get the spin configurations.
We evaluate the energy of the samples to get the energy distribution, and plot the distribution in @fig:ising-energy-distribution.

#figure(image("../chap3-optimization/images/ising-energy-distribution.svg", width: 80%, alt: "Ising energy distribution"),
caption: [The binned energy distribution of spin configurations generated unbiasly from the ferromagnetic Ising model ($J_(i j) = -1, L = 10$) at different inverse temperatures $beta$. The method to generate the samples is the tensor network based method detailed in @Roa2024]
) <fig:ising-energy-distribution>

To characterize a phase transition, physicists prefers to define an *order parameter*, which is a quantity that is non-zero in the ordered phase and zero in the disordered phase.
For the ferromagnetic Ising model, the order parameter is the magnetization, which is defined as
$ |m| = lr(|sum_(i=1)^(L^2) s_i\/L^2|). $ <eq:magnetization>
We are interested in the statistical average of $|m|$ over the configuration space at different temperatures,
$
angle.l |m| angle.r_beta = sum_(bold(s) in S) |m(bold(s))| p_beta (bold(s)).
$ <eq:magnetization-average>
Since the ground state of the ferromagnetic Ising model is two fold degenerate, all-up and all-down, at zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r_infinity = 1$.

We can use #link("https://github.com/QuEraComputing/GenericTensorNetworks.jl")[GenericTensorNetworks.jl]@Liu2023 to analyse the solution space properties, such as the ground state degeneracy and the connectivity of the solution space.
```julia
julia> using GenericTensorNetworks

julia> graph = grid((6, 6))
julia> problem = SpinGlass(graph, -ones(Int, ne(graph)), zeros(Int, nv(graph)));

julia> solve(problem, ConfigsMin())[]  # solve the spin glass ground state
(-60.0, {000000000000000000000000000000000000, 111111111111111111111111111111111111})ₜ

julia> solutionspace = solve(problem, ConfigsMin(9))[];  # States with energy E0 ~ E0+8
julia> show_landscape((x, y)->hamming_distance(x, y) <= 1, solutionspace; layer_distance=-30, optimal_distance=20, filename="grid66.svg");
```
The returned `solutionspace` object contains the lowest energy and the associated configurations.
We can see the ground states are two fold degenerate, they are the all-up and all-down configurations.
The `show_landscape` function visualizes the energy landscape (@fig:ising-energy-landscape), and the configurations are connected if they differ by flipping a single spin.
#figure(image("images/grid66.svg", width: 80%, alt: "Energy landscape"),
caption: [The energy landscape of low energy states of the ferromagnetic Ising model on a 6x6 grid. The two degenerate ground states are the all-up and all-down configurations. Configurations are connected if they differ by flipping a single spin.]
) <fig:ising-energy-landscape>

#figure(
  canvas(length: 0.7cm, {
  import plot: *
  import draw: *
  let f1 = x=>calc.pow(x, 2)
  plot(
    size: (8,6),
    x-tick-step: none,
    y-tick-step: none,
    x-label: [$bold(n)$],
    y-label: [$H(bold(n))$],
    name: "plot",
    {
      add(domain: (-2, 2), f1)
      add-fill-between(domain:(-2, 2), x=>calc.min(f1(x), 0.6), x=>0.6)
      add-anchor("p1", (-0.7,f1(-0.7)))
      add-anchor("p2", (0.7,f1(0.7)))
      add-anchor("p3", (-0.5,f1(-0.5)))
      add-anchor("p4", (0.5,f1(0.5)))
    }
  )
  line("plot.p1", "plot.p2", mark: (end: "straight"), stroke:black, name:"line1")
  content(("line1.start", 1.5, "line1.end"),
    anchor: "south",
    padding: 10pt,
    [Without OGP]
  )
  line("plot.p3", "plot.p4", mark: (end: "straight"), stroke:black)
  content((-1, 1), [$-gamma alpha(G)$])

  set-origin((0, -4))
  let f3 = x=>{
    if (x < 0.4) {
      return 1-calc.pow(x - 0.2, 2) * 25
    } else {
      return 0
    }
  }
  plot(
    size: (8,3),
    x-tick-step: none,
    y-tick-step: none,
    x-label: "Hamming distance",
    y-label: "Count",
    name: "plot",
    {
      add(domain: (0, 1), f3, fill:true)
    }
  )

  set-origin((10, 4))
  let f2 = x=>calc.cos(3*x+4*calc.pow(x, 2))
  plot(
    size: (8,6),
    x-tick-step: none,
    y-tick-step: none,
    x-label: [$bold(n)$],
    y-label: [$H(bold(n))$],
    name: "plot",
    {
      add(domain: (-2, 2), f2, samples: 200)
      add-fill-between(domain:(-2, 2), x=>calc.min(f2(x), -0.7), x=>-0.7, samples: 200)
      add-anchor("p3", (-1.4,f2(-1.4)))
      add-anchor("p4", (0.5,f2(-1.4)))
      add-anchor("p1", (-1.4,f2(-1.25)))
      add-anchor("p2", (-1.25,f2(-1.25)))
    }
  )
  line("plot.p1", "plot.p2", mark: (end: "straight"), stroke:black, name:"line1")
  content(("line1.start", 1.8, "line1.end"),
    anchor: "south",
    padding: 10pt,
    [With OGP]
  )
  line("plot.p3", "plot.p4", mark: (end: "straight"), stroke:black)

  set-origin((0, -4))
  let f3 = x=>{
    if (x < 0.2) {
      return 1-calc.pow(x - 0.1, 2) * 100
    } else if (x > 0.5 and x < 0.7) {
      return 1-calc.pow(x - 0.6, 2) * 100
    } else {
      return 0
    }
  }
  plot(
    size: (8,3),
    x-tick-step: none,
    y-tick-step: none,
    x-label: "Hamming distance",
    y-label: "Count",
    name: "plot",
    {
      add(domain: (0, 1), f3, fill:true)
    }
  )
  content((2.5, 1.6), text(12pt)[intra valley])
  content((6.5, 1.6), text(12pt)[inter valley])

  }),
  caption: [The different types of energy landscape revealed by the overlap gap property (OGP)@Gamarnik2021. The left panel show the energy landscape without OGP, the right panel show the energy landscape with OGP.]
)
== Integrate a function with Monte Carlo method

Tensor network based methods have its limitation on the size of the system. For system sizes larger than $30 times 30$, the tensor network based method is infeasible.
Instead, we resort to the Monte Carlo method to sample a portion of the configuration space and estimate the statistical average of the quantity we are interested in.

Monte Carlo method can be viewed as a efficient way to estimate the integral of a function over a domain. We consider integrating a positive function $f(x)$ defined on a unit square.
$
integral f(x) d x
$
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
To evaluate the @eq:magnetization-average, we can sample the configuration space with the Boltzmann distribution $p(bold(s)) = (e^(-beta H(bold(s))))\/Z$, on each sample, we evaluate $|m(bold(s))|$ and calculate the statistical average.
The problem is that we do not know the partition function $Z$. Even if we know how to compute $p(bold(s))$, the sampling is still challenging.
Metropolis-Hastings algorithm is a sampling method to sample the configuration space with un-normalized probability.

The Metropolis-Hastings algorithm is a Markov chain. A Markov chain is a sequence of random variables $bold(s)_1, bold(s)_2, dots$ with the property that the probability of moving to the next state depends only on the current state.
It is characterized by the transition probability $P(bold(s)'|bold(s))$, the probability of moving from $bold(s)$ to $bold(s)'$.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  let dx = 2.8
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
  Function([Metropolis-Hastings], ([$beta$], [$n$]), {
    Assign([$bold(s)$], [Initial a random configuration])
    For($i = 1, dots, n$, {
      Assign([$bold(s)'$], [propose a new configuration with prior probability $T(bold(s)arrow.r bold(s)')$])
      Assign([$bold(s)$], [$bold(s)' "with probability:" min(1, (T(bold(s)' arrow.r bold(s)) p(bold(s)'))/(T(bold(s) arrow.r bold(s)') p(bold(s))))$])
    })
  })
 })

In line 4, a new configuration $bold(s)'$ is proposed. The probability of proposing $bold(s)'$ from $bold(s)$ is $T(bold(s)arrow.r bold(s)')$, which is known when we design the algorithm. For example, when we propose a state transfer on two 3-state spins, we may have the following transition rule:
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

== Simulation results
The following results are obtained from the simple update MCMC detailed in the code demo: #link("https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/IsingModel")[IsingModel].
They correspond to the update of spin configurations at temperature $T = 1$ and $T = 3$ respectively, and the critical temperature $T_c =2/ln(1+sqrt(2)) approx 2.269$.
Below $T_c$, the system tend to form a large cluster of spins in the same direction, the magnetization is non-zero. Above $T_c$, the system is in the paramagnetic phase, the magnetization is zero.
#grid(columns: 2, column-gutter: 50pt, figure(image("images/ising-spins-1.0.gif", width: 100%, alt: "Ising model at T = 1")), figure(image("images/ising-spins-3.0.gif", width: 100%, alt: "Ising model at T = 3")), align(center)[#h(4em)$T = 1 (<T_c)$], align(center)[#h(4em)$T = 3 (>T_c)$])

#figure(image("images/ising-data.png", width: 80%, alt: "Ising model data"),
caption: [(a) the observable values for the Ising model at temperature $T = 1$ using the simple update MCMC. Each sweep processes to update every spin once, and the observable is averaged over 1000 samples. (b) the autocorrelation function as a function of the lag $tau$, the inverse slope of the line is the autocorrelation time $Theta$.]
) <fig:ising-data>

However, the simple update does not thermalize fast enough. In the ferromagnetic phase, the domain wall grows too slow to reach the whole system.
As shown in @fig:ising-data (b), the *autocorrelation time* is very long, indicating a strong correlation between the samples. Here, the autocorrelation time, $Theta$, is the number of steps it takes for the correlation between two consecutive samples to decay to one half of the maximum correlation:
$
  A_Q (tau) = (angle.l Q_k Q_(k+tau) angle.r - angle.l Q_k angle.r^2)/(angle.l Q_k^2 angle.r - angle.l Q_k angle.r^2).
$
where $Q_k$ is the quantity of interest evaluated at step $k$. The autocorrelation function is strictly $<=1$, and the smaller it is, the less correlated the samples are. When the samples are independent, we have $angle.l Q_k Q_(k+tau) angle.r = angle.l Q_k angle.r angle.l Q_(k+tau) angle.r arrow.double.r A_Q (tau) = 0$. A typical behavior is $A_Q (tau) ~ e^(-tau\/Theta)$.

Instead of flip one spin a time, we can update a cluster of spins at a time. The cluster update proposed in @Swendsen1987 is a good example.
This method improves both the *acceptance rate* and the *autocorrelation time*.

The following results are obtained from the cluster update MCMC. We can see the system thermalizes in a few tens of sweeps.
#grid(columns: 2, column-gutter: 50pt, figure(image("images/swising-spins-1.0.gif", width: 100%, alt: "Ising model at T = 1")), figure(image("images/swising-spins-3.0.gif", width: 100%, alt: "Ising model at T = 3")), align(center)[#h(4em)$T = 1 (<T_c)$], align(center)[#h(4em)$T = 3 (>T_c)$])

#figure(image("images/sw-data.png", width: 80%, alt: "Ising model data"),
caption: [The same observables as in @fig:ising-data, but using the cluster update MCMC. The autocorrelation time below $10^(-3)$ are not shown.]
) <fig:sw-data>

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

# initilaize a 6 site Ising model on a cycle
graph = Graphs.cycle_graph(6)
model = SpinGlass(graph, -ones(ne(graph)), zeros(nv(graph)))
for T in [0.5, 1.0, 2.0, 4.0, 10.0]
    gap = spectral_gap(transition_matrix(model, 1/T))
    println("T: $T, Spectral gap: $gap")
end    
# output:
# T: 0.5, Spectral gap: 0.0001676924058030549
# T: 1.0, Spectral gap: 0.009043491536947279
# T: 2.0, Spectral gap: 0.062025555293211854
# T: 4.0, Spectral gap: 0.14992147959036217
# T: 10.0, Spectral gap: 0.24477825057138192
```

We plot the output above in the following figure, where we can see a clear trend of exponential decay of the spectral gap with the inverse temperature.

#figure(image("images/spectralgap.svg", width: 300pt, alt: "Spectral gap"),
caption: [Spectral gap v.s. $1\/T$ of the Ising model ($J = -1$) on a circle of length $N=6$.],
)

== Cheeger's inequality

Cheeger's inequality is a fundamental result in spectral graph theory that relates the conductance (or isoperimetric constant) of a graph to its spectral gap. This relationship is particularly important in the context of spin glass systems and Markov Chain Monte Carlo methods, as it provides bounds on mixing times.

=== Conductance and the Cheeger constant

For a graph $G = (V, E)$ with vertex set $V$ and edge set $E$, the Cheeger constant (or conductance) $h(G)$ is defined as the probability of escaping from the most inescapable set:

$
h(G) = min_(S subset V, 0 < |S| <= (|V|)/2) frac(|E(S, V backslash S)|, min("vol"(S), "vol"(V backslash S)))
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

The mixing time of a Markov chain is the time required for the chain to approach its stationary distribution. For a reversible Markov chain, the mixing time $t_"mix"$ is related to the spectral gap $(1 - lambda_2)$ of the transition matrix: $t_"mix" approx frac(1, 1 - lambda_2)$.
By Cheeger's inequality, we know that:

$
1 - lambda_2 >= frac(h(G)^2, 2)
$

Therefore:

$
t_"mix" <= frac(2, h(G)^2)
$

This means that a graph with a large Cheeger constant (good expansion properties) will have a small mixing time, allowing MCMC methods to converge quickly to the stationary distribution.

In practice, for spin glass systems, the Cheeger constant provides a quantitative measure of how "glassy" the energy landscape is. Systems with small Cheeger constants have energy landscapes with high barriers between different metastable states, making equilibration difficult and necessitating techniques like parallel tempering to efficiently sample the state space.

#bibliography("refs.bib")