#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#import "@preview/ctheorems:1.1.3": *

#show: book-page.with(title: "Spin Glass")
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

// Usage example:
#let random_numbers = pseudo_random(12345, 100)
#let random_bools(p) = random_numbers.map(x => x < p)

#align(center, [= Spin Glass\
_Jin-Guo Liu_])



== Main references
- Lecture note: #link("https://physics.bu.edu/~py502/lectures5/mc.pdf")[Monte Carlo simulations in classical statistical physics, Anders Sandvik]
- Swendsen, Robert H., and Jian-Sheng Wang. "Nonuniversal critical dynamics in Monte Carlo simulations." Physical review letters 58.2 (1987): 86.

== Ising model

An Ising model is defined by a graph $G = (V, E)$ and a set of spins $s_i in {-1, 1}$ for each vertex $i in V$. The energy of the system is given by

$
H = - sum_((i,j) in E) J_(i j) s_i s_j - sum_(i in V) h_i s_i
$

We are interested in Ising models for multiple reasons, one is from the physics perspective. We want to understand the phenomemon of phase transition, i.e. how magnetization emerges from the disorder. Another reason is from the optimization perspective, finding the ground state (the configuration with the lowest energy) of an Ising model is in general hard, which is known as the spin glass problem. It is in complexity class NP-complete. Solving the spin glass problem in time polynomial to the graph size would break the complexity hierarchy, which is unlikely to happen.

== Phase transition

In this section, we consider the ferromagnetic Ising model, where $J_(i j) = 1$. For simplicity, we limit our discussion to the $L times L$ grid.
We denote the solution space as $S$, and $bold(s) in S$ is the configuration of spins, e.g. $bold(s) = {-1, -1, 1, -1, 1, dots}$. The number of configurations is $|S| = 2^(L^2)$, which is exponentially large.

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
  spinconfig(4, 4, random_bools(0.5).slice(4, 20))
  content((5, 1.5), s[$|S| = 2^(L^2)$])
}))

When we talk about the phase transition, we are interested in how does the macroscopic properties of the system change as the temperature $T$ changes, like how ice becomes water, magnets lose their magnetization, etc.
Computation is a valid strategy to study the phase transition. Before we move on, we need to consider the following question:
- How to describe the system?
- How to characterize the phase transition?

At finite temperature, the system can be described by a probability distribution, called the Boltzmann distribution:
$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,
$
where $beta = 1 \/ k_B T$ is the inverse temperature, $Z$ is the partition function $Z = sum_bold(s) e^(-beta H(bold(s)))$ that normalizes the probability distribution.
Configurations with lower energy have higher probability to be observed. The sensitivity of the probability distribution to the energy is determined by the inverse temperature $beta$. At zero temperature, $beta$ is infinite, and only the ground state can be observed.

We characterize the macroscopic properties of the system by the statistical average of some functions over the configuration space. Among these functions, the *order parameter* can be used to characterize the phase transition.
As shown in @fig:phase-transition, the order parameter for the magnetization can be defined as $ |m| = lr(|sum_(i=1)^(L^2) s_i\/L^2|). $
At the inifinite size limit ($L arrow.r infinity$), if the statistical average $angle.l |m| angle.r$ is non-zero, the system is in the magnetized phase. If $angle.l |m| angle.r$ is zero, the system is in the disordered phase.
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

For the ferromagnetic Ising model, the ground state is two fold degenerate, they are the all-up and all-down configurations. At zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r = 1$.

== Physical quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$

== The spirit of importance sampling

Given a positive function $f(x)$, we want to calculate the integral
$
integral f(x) d x
$
Intead of evaluating the integral directly, the importance sampling can sample $x$ with probability $p(x)$ and estimate the integral as the statistical average of $f(x)\/p(x)$. To see how it works, we consider the example in @fig:importance-sampling.
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

The function is a peak-like function defined on a unit square, which is zero everywhere except a very small region (the red circle of radius $r << 1$) at the origin. The integral of $f(x)$ is 1. If your sampling is uniform, you will spend most of the time sampling the region far away from the origin.
However, if you sample $x$ with $10 times$ more probability near the origin (in the dark blue region), as shown in the right panel, you can have $10 times$ more chance to find a sample in the peak region. So, whenever you find a sample in the dark blue region, you only count it as $0.1$ sample as a compensation. With this small change, the statistical average will be $sqrt(10)$ times more accurate. In a even more extreme case, if the sampled probability $p(x)$ is proportional to $f(x)$, the number of sample to reach exact result is $1$.

For bad sampling probability $p(x)$, the function $f(x)$ may never have a good estimate even if you have infinite samples. This happens when the *ergodicity* is broken, i.e. the system can not reach the whole configuration space with non-zero probability. For example, in @fig:ergodicity, the function $f(x)$ has two peaks, but only one peak is accessible to the sampler.
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(100%))
  rect((-1.5, -1.5), (0.5, 0.5), stroke: none, fill: blue.lighten(50%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  circle((0.8, 0.8), radius: 0.1, fill: red, stroke: none)
  content((0, -2.5), s[Ergodicity is broken])
  content((-0.5, -0.7), s[Accessible\ region])
})) <fig:ergodicity>


== Goal

Sample with probability $p(s)$ to calculate the physical quantities.

== Markov chain

A sequence of random variables $X_1, X_2, dots$ with the property that the probability of moving to the next state depends only on the current state.

The transition probability is $P(X_(t+1) = x' | X_t = x)$. The transition matrix is $P_(i j) = P(X_(t+1) = x_j | X_t = x_i)$.

== Two key properties
1. Ergodicity: the system can reach any state in the configuration space
2. Detailed balance: the probability of going from $s$ to $s'$ is the same as going from $s'$ to $s$

== Metropolis algorithm

Acceptance probability: $p = min(1, e^(-beta Delta H))$

== Monte Carlo simulation
1. Initialize the system to a random configuration
2. Repeat the following steps:
   - Randomly flip a spin
   - Accept the new configuration with probability $p = min(1, e^(-beta Delta H))$
   - If the system is thermalized, calculate the physical quantities

== Demonstration

== Hands on!